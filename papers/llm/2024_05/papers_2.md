# llm - 2024_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06549v1">Large Language Model (LLM) for Standard Cell Layout Design Optimization</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 6 pages, 8 figures, IEEE International Workshop on LLM-Aided Design (LAD'24)
    </div>
    <details class="paper-abstract">
      Standard cells are essential components of modern digital circuit designs. With process technologies advancing toward 2nm, more routability issues have arisen due to the decreasing number of routing tracks, increasing number and complexity of design rules, and strict patterning rules. The state-of-the-art standard cell design automation framework is able to automatically design standard cell layouts in advanced nodes, but it is still struggling to generate highly competitive Performance-Power-Area (PPA) and routable cell layouts for complex sequential cell designs. Consequently, a novel and efficient methodology incorporating the expertise of experienced human designers to incrementally optimize the PPA of cell layouts is highly necessary and essential. High-quality device clustering, with consideration of netlist topology, diffusion sharing/break and routability in the layouts, can reduce complexity and assist in finding highly competitive PPA, and routable layouts faster. In this paper, we leverage the natural language and reasoning ability of Large Language Model (LLM) to generate high-quality cluster constraints incrementally to optimize the cell layout PPA and debug the routability with ReAct prompting. On a benchmark of sequential standard cells in 2nm, we demonstrate that the proposed method not only achieves up to 19.4% smaller cell area, but also generates 23.5% more LVS/DRC clean cell layouts than previous work. In summary, the proposed method not only successfully reduces cell area by 4.65% on average, but also is able to fix routability in the cell layout designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15202v1">Cross-Task Defense: Instruction-Tuning LLMs for Content Safety</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 accepted to NAACL2024 TrustNLP workshop
    </div>
    <details class="paper-abstract">
      Recent studies reveal that Large Language Models (LLMs) face challenges in balancing safety with utility, particularly when processing long texts for NLP tasks like summarization and translation. Despite defenses against malicious short questions, the ability of LLMs to safely handle dangerous long content, such as manuals teaching illicit activities, remains unclear. Our work aims to develop robust defenses for LLMs in processing malicious documents alongside benign NLP task queries. We introduce a defense dataset comprised of safety-related examples and propose single-task and mixed-task losses for instruction tuning. Our empirical results demonstrate that LLMs can significantly enhance their capacity to safely manage dangerous content with appropriate instruction tuning. Additionally, strengthening the defenses of tasks most susceptible to misuse is effective in protecting LLMs against processing harmful information. We also observe that trade-offs between utility and safety exist in defense strategies, where Llama2, utilizing our proposed approach, displays a significantly better balance compared to Llama1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15165v1">A Solution-based LLM API-using Methodology for Academic Information Seeking</a></div>
    <div class="paper-meta">
      📅 2024-05-24
      | 💬 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Applying large language models (LLMs) for academic API usage shows promise in reducing researchers' academic information seeking efforts. However, current LLM API-using methods struggle with complex API coupling commonly encountered in academic queries. To address this, we introduce SoAy, a solution-based LLM API-using methodology for academic information seeking. It uses code with a solution as the reasoning method, where a solution is a pre-constructed API calling sequence. The addition of the solution reduces the difficulty for the model to understand the complex relationships between APIs. Code improves the efficiency of reasoning. To evaluate SoAy, we introduce SoAyBench, an evaluation benchmark accompanied by SoAyEval, built upon a cloned environment of APIs from AMiner. Experimental results demonstrate a 34.58-75.99\% performance improvement compared to state-of-the-art LLM API-based baselines. All datasets, codes, tuned models, and deployed online services are publicly accessible at https://github.com/RUCKBReasoning/SoAy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15114v1">Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning</a></div>
    <div class="paper-meta">
      📅 2024-05-24
    </div>
    <details class="paper-abstract">
      Conventional recommender systems (RSs) face challenges in precisely capturing users' fine-grained preferences. Large language models (LLMs) have shown capabilities in commonsense reasoning and leveraging external tools that may help address these challenges. However, existing LLM-based RSs suffer from hallucinations, misalignment between the semantic space of items and the behavior space of users, or overly simplistic control strategies (e.g., whether to rank or directly present existing results). To bridge these gap, we introduce ToolRec, a framework for LLM-empowered recommendations via tool learning that uses LLMs as surrogate users, thereby guiding the recommendation process and invoking external tools to generate a recommendation list that aligns closely with users' nuanced preferences. We formulate the recommendation process as a process aimed at exploring user interests in attribute granularity. The process factors in the nuances of the context and user preferences. The LLM then invokes external tools based on a user's attribute instructions and probes different segments of the item pool. We consider two types of attribute-oriented tools: rank tools and retrieval tools. Through the integration of LLMs, ToolRec enables conventional recommender systems to become external tools with a natural language interface. Extensive experiments verify the effectiveness of ToolRec, particularly in scenarios that are rich in semantic content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14748v1">MultiCast: Zero-Shot Multivariate Time Series Forecasting Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Predicting future values in multivariate time series is vital across various domains. This work explores the use of large language models (LLMs) for this task. However, LLMs typically handle one-dimensional data. We introduce MultiCast, a zero-shot LLM-based approach for multivariate time series forecasting. It allows LLMs to receive multivariate time series as input, through three novel token multiplexing solutions that effectively reduce dimensionality while preserving key repetitive patterns. Additionally, a quantization scheme helps LLMs to better learn these patterns, while significantly reducing token use for practical applications. We showcase the performance of our approach in terms of RMSE and execution time against state-of-the-art approaches on three real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14636v1">PerLLM: Personalized Inference Scheduling with Edge-Cloud Collaboration for Diverse LLM Services</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      With the rapid growth in the number of large language model (LLM) users, it is difficult for bandwidth-constrained cloud servers to simultaneously process massive LLM services in real-time. Recently, edge-cloud infrastructures have been used to improve the processing efficiency of large-scale LLM services. However, the diversity of task requirements and the dynamics of resources pose great challenges to inference scheduling, leading to the wastage of many resources. In this paper, we present PerLLM, a personalized inference scheduling framework with edge-cloud collaboration designed for diverse LLM services. For the complexity of multiple constraints and the decision-making process of edge-cloud collaboration, we integrate the upper confidence bound algorithm based on the constraint satisfaction mechanism in PerLLM. For diverse LLM services, PerLLM can optimize service scheduling and resource allocation solutions within the edge-cloud infrastructure to meet processing time requirements while minimizing energy costs. Experimental results from different model deployments show that PerLLM can effectively meet the processing time requirements of personalized services. Compared to other methods, PerLLM achieves 2.2x, 2.1x, and 1.6x throughput and reduces the energy cost by more than 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02030v2">Panacea: Pareto Alignment via Preference Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Current methods for large language model alignment typically use scalar human preference labels. However, this convention tends to oversimplify the multi-dimensional and heterogeneous nature of human preferences, leading to reduced expressivity and even misalignment. This paper presents Panacea, an innovative approach that reframes alignment as a multi-dimensional preference optimization problem. Panacea trains a single model capable of adapting online and Pareto-optimally to diverse sets of preferences without the need for further tuning. A major challenge here is using a low-dimensional preference vector to guide the model's behavior, despite it being governed by an overwhelmingly large number of parameters. To address this, Panacea is designed to use singular value decomposition (SVD)-based low-rank adaptation, which allows the preference vector to be simply injected online as singular values. Theoretically, we prove that Panacea recovers the entire Pareto front with common loss aggregation methods under mild conditions. Moreover, our experiments demonstrate, for the first time, the feasibility of aligning a single LLM to represent an exponentially vast spectrum of human preferences through various optimization methods. Our work marks a step forward in effectively and efficiently aligning models to diverse and intricate human preferences in a controllable and Pareto-optimal manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17752v3">Can multiple-choice questions really be useful in detecting the abilities of LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 LREC-COLING 2024
    </div>
    <details class="paper-abstract">
      Multiple-choice questions (MCQs) are widely used in the evaluation of large language models (LLMs) due to their simplicity and efficiency. However, there are concerns about whether MCQs can truly measure LLM's capabilities, particularly in knowledge-intensive scenarios where long-form generation (LFG) answers are required. The misalignment between the task and the evaluation method demands a thoughtful analysis of MCQ's efficacy, which we undertake in this paper by evaluating nine LLMs on four question-answering (QA) datasets in two languages: Chinese and English. We identify a significant issue: LLMs exhibit an order sensitivity in bilingual MCQs, favoring answers located at specific positions, i.e., the first position. We further quantify the gap between MCQs and long-form generation questions (LFGQs) by comparing their direct outputs, token logits, and embeddings. Our results reveal a relatively low correlation between answers from MCQs and LFGQs for identical questions. Additionally, we propose two methods to quantify the consistency and confidence of LLMs' output, which can be generalized to other QA evaluation benchmarks. Notably, our analysis challenges the idea that the higher the consistency, the greater the accuracy. We also find MCQs to be less reliable than LFGQs in terms of expected calibration error. Finally, the misalignment between MCQs and LFGQs is not only reflected in the evaluation performance but also in the embedding space. Our code and models can be accessed at https://github.com/Meetyou-AI-Lab/Can-MC-Evaluate-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14488v1">MoGU: A Framework for Enhancing Safety of Open-Sourced LLMs While Preserving Their Usability</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in various applications. As their usage grows, concerns regarding their safety are rising, especially in maintaining harmless responses when faced with malicious instructions. Many defense strategies have been developed to enhance the safety of LLMs. However, our research finds that existing defense strategies lead LLMs to predominantly adopt a rejection-oriented stance, thereby diminishing the usability of their responses to benign instructions. To solve this problem, we introduce the MoGU framework, designed to enhance LLMs' safety while preserving their usability. Our MoGU framework transforms the base LLM into two variants: the usable LLM and the safe LLM, and further employs dynamic routing to balance their contribution. When encountering malicious instructions, the router will assign a higher weight to the safe LLM to ensure that responses are harmless. Conversely, for benign instructions, the router prioritizes the usable LLM, facilitating usable and helpful responses. On various open-sourced LLMs, we compare multiple defense strategies to verify the superiority of our MoGU framework. Besides, our analysis provides key insights into the effectiveness of MoGU and verifies that our designed routing mechanism can effectively balance the contribution of each variant by assigning weights. Our work released the safer Llama2, Vicuna, Falcon, Dolphin, and Baichuan2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14487v1">A Comprehensive Overview of Large Language Models (LLMs) for Cyber Defences: Opportunities and Directions</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      The recent progression of Large Language Models (LLMs) has witnessed great success in the fields of data-centric applications. LLMs trained on massive textual datasets showed ability to encode not only context but also ability to provide powerful comprehension to downstream tasks. Interestingly, Generative Pre-trained Transformers utilised this ability to bring AI a step closer to human being replacement in at least datacentric applications. Such power can be leveraged to identify anomalies of cyber threats, enhance incident response, and automate routine security operations. We provide an overview for the recent activities of LLMs in cyber defence sections, as well as categorization for the cyber defence sections such as threat intelligence, vulnerability assessment, network security, privacy preserving, awareness and training, automation, and ethical guidelines. Fundamental concepts of the progression of LLMs from Transformers, Pre-trained Transformers, and GPT is presented. Next, the recent works of each section is surveyed with the related strengths and weaknesses. A special section about the challenges and directions of LLMs in cyber security is provided. Finally, possible future research directions for benefiting from LLMs in cyber security is discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14428v1">Mitigating Quantization Errors Due to Activation Spikes in GLU-Based LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) have established state-of-the-art performance through architectural improvements, but still require significant computational cost for inference. In an effort to reduce the inference cost, post-training quantization (PTQ) has become a popular approach, quantizing weights and activations to lower precision, such as INT8. In this paper, we reveal the challenges of activation quantization in GLU variants, which are widely used in feed-forward network (FFN) of modern LLMs, such as LLaMA family. The problem is that severe local quantization errors, caused by excessive magnitudes of activation in GLU variants, significantly degrade the performance of the quantized LLM. We denote these activations as activation spikes. Our further observations provide a systematic pattern of activation spikes: 1) The activation spikes occur in the FFN of specific layers, particularly in the early and late layers, 2) The activation spikes are dedicated to a couple of tokens, rather than being shared across a sequence. Based on our observations, we propose two empirical methods, Quantization-free Module (QFeM) and Quantization-free Prefix (QFeP), to isolate the activation spikes during quantization. Our extensive experiments validate the effectiveness of the proposed methods for the activation quantization, especially with coarse-grained scheme, of latest LLMs with GLU variants, including LLaMA-2/3, Mistral, Mixtral, SOLAR, and Gemma. In particular, our methods enhance the current alleviation techniques (e.g., SmoothQuant) that fail to control the activation spikes. Code is available at https://github.com/onnoo/activation-spikes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14390v1">Speculating About Multi-user Conversational Interfaces and LLMs: What If Chatting Wasn't So Lonely?</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 To appear in the proceedings of the 2024 ACM Conference on Conversational User Interfaces (CUI 24)
    </div>
    <details class="paper-abstract">
      The advent of LLMs means that CUIs are cool again, but what isn't so cool is that we're doomed to use them alone. The one user, one account, one device paradigm has dominated the design of CUIs and is not going away as new conversational technologies emerge. In this provocation we explore some of the technical, legal, and design difficulties that seem to make multi-user CUIs so difficult to implement. Drawing inspiration from the ways that people manage messy group discussions, such as parliamentary and consensus-based paradigms, we show how LLM-based CUIs might be well suited to bridging the gap. With any luck, this might even result in everyone having to sit through fewer poorly run meetings and agonising group discussions - truly a laudable goal!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14371v1">EdgeShard: Efficient LLM Inference via Collaborative Edge Computing</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential in natural language processing and content generation. However, current LLMs heavily rely on cloud computing, leading to prolonged latency, high bandwidth cost, and privacy concerns. Edge computing is promising to address such concerns by deploying LLMs on edge devices, closer to data sources. Some works try to leverage model quantization to reduce the model size to fit the resource-constraint edge devices, but they lead to accuracy loss. Other works use cloud-edge collaboration, suffering from unstable network connections. In this work, we leverage collaborative edge computing to facilitate the collaboration among edge devices and cloud servers for jointly performing efficient LLM inference. We propose a general framework to partition the LLM model into shards and deploy on distributed devices. To achieve efficient LLM inference, we formulate an adaptive joint device selection and model partition problem and design an efficient dynamic programming algorithm to optimize the inference latency and throughput, respectively. Experiments of Llama2 serial models on a heterogeneous physical prototype demonstrate that EdgeShard achieves up to 50% latency reduction and 2x throughput improvement over baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14333v1">DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Proof assistants like Lean have revolutionized mathematical proof verification, ensuring high accuracy and reliability. Although large language models (LLMs) show promise in mathematical reasoning, their advancement in formal theorem proving is hindered by a lack of training data. To address this issue, we introduce an approach to generate extensive Lean 4 proof data derived from high-school and undergraduate-level mathematical competition problems. This approach involves translating natural language problems into formal statements, filtering out low-quality statements, and generating proofs to create synthetic data. After fine-tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof generation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%. Additionally, our model successfully proved 5 out of 148 problems in the Lean 4 Formalized International Mathematical Olympiad (FIMO) benchmark, while GPT-4 failed to prove any. These results demonstrate the potential of leveraging large-scale synthetic data to enhance theorem-proving capabilities in LLMs. Both the synthetic dataset and the model will be made available to facilitate further research in this promising field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14231v1">From Role-Play to Drama-Interaction: An LLM Solution</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 Accepted by ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Drama is a form of storytelling inspired by human creativity, proceeding with a predefined storyline, carrying emotions and thoughts. This paper introduces \emph{LLM-based interactive drama}, which endows traditional drama with an unprecedented immersion, where a person is allowed to walk into it and interact with the characters and scenes. We define this new artistic genre by 6 essential elements-plot, character, thought, diction, spectacle and interaction-and study the entire pipeline to forge a backbone \emph{drama LLM} to drive the playing process, which is challenged by limited drama resources, uncontrollable narrative development, and complicated instruction following. We propose \emph{Narrative Chain} to offer finer control over the narrative progression during interaction with players; \emph{Auto-Drama} to synthesize drama scripts given arbitrary stories; \emph{Sparse Instruction Tuning} to allow the model to follow sophisticated instructions. We manually craft 3 scripts, \emph{Detective Conan}, \emph{Harry Potter}, \emph{Romeo and Juliet}, and design a 5-dimension principle to evaluate the drama LLM comprehensively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12482v2">Embodied LLM Agents Learn to Cooperate in Organized Teams</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as integral tools for reasoning, planning, and decision-making, drawing upon their extensive world knowledge and proficiency in language-related tasks. LLMs thus hold tremendous potential for natural language interaction within multi-agent systems to foster cooperation. However, LLM agents tend to over-report and comply with any instruction, which may result in information redundancy and confusion in multi-agent cooperation. Inspired by human organizations, this paper introduces a framework that imposes prompt-based organization structures on LLM agents to mitigate these problems. Through a series of experiments with embodied LLM agents and human-agent collaboration, our results highlight the impact of designated leadership on team efficiency, shedding light on the leadership qualities displayed by LLM agents and their spontaneous cooperative behaviors. Further, we harness the potential of LLMs to propose enhanced organizational prompts, via a Criticize-Reflect process, resulting in novel organization structures that reduce communication costs and enhance team efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14189v1">Semantic-guided Prompt Organization for Universal Goal Hijacking against LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      With the rising popularity of Large Language Models (LLMs), assessing their trustworthiness through security tasks has gained critical importance. Regarding the new task of universal goal hijacking, previous efforts have concentrated solely on optimization algorithms, overlooking the crucial role of the prompt. To fill this gap, we propose a universal goal hijacking method called POUGH that incorporates semantic-guided prompt processing strategies. Specifically, the method starts with a sampling strategy to select representative prompts from a candidate pool, followed by a ranking strategy that prioritizes the prompts. Once the prompts are organized sequentially, the method employs an iterative optimization algorithm to generate the universal fixed suffix for the prompts. Experiments conducted on four popular LLMs and ten types of target responses verified the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14169v1">Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 12 pages, 5 tables, 5 figures, work in progress
    </div>
    <details class="paper-abstract">
      Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07300v2">CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Deep learning (e.g., Transformer) has been widely and successfully used in multivariate time series forecasting (MTSF). Unlike existing methods that focus on training models from a single modal of time series input, large language models (LLMs) based MTSF methods with cross-modal text and time series input have recently shown great superiority, especially with limited temporal data. However, current LLM-based MTSF methods usually focus on adapting and fine-tuning LLMs, while neglecting the distribution discrepancy between textual and temporal input tokens, thus leading to sub-optimal performance. To address this issue, we propose a novel Cross-Modal LLM Fine-Tuning (CALF) framework for MTSF by reducing the distribution discrepancy between textual and temporal data, which mainly consists of the temporal target branch with temporal input and the textual source branch with aligned textual input. To reduce the distribution discrepancy, we develop the cross-modal match module to first align cross-modal input distributions. Additionally, to minimize the modality distribution gap in both feature and output spaces, feature regularization loss is developed to align the intermediate features between the two branches for better weight updates, while output consistency loss is introduced to allow the output representations of both branches to correspond effectively. Thanks to the modality alignment, CALF establishes state-of-the-art performance for both long-term and short-term forecasting tasks with low computational complexity, and exhibiting favorable few-shot and zero-shot abilities similar to that in LLMs. Code is available at \url{https://github.com/Hank0626/LLaTA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04853v2">Leveraging LLMs for Unsupervised Dense Retriever Ranking</a></div>
    <div class="paper-meta">
      📅 2024-05-23
      | 💬 SIGIR2024 full paper
    </div>
    <details class="paper-abstract">
      In this paper we present Large Language Model Assisted Retrieval Model Ranking (LARMOR), an effective unsupervised approach that leverages LLMs for selecting which dense retriever to use on a test corpus (target). Dense retriever selection is crucial for many IR applications that rely on using dense retrievers trained on public corpora to encode or search a new, private target corpus. This is because when confronted with domain shift, where the downstream corpora, domains, or tasks of the target corpus differ from the domain/task the dense retriever was trained on, its performance often drops. Furthermore, when the target corpus is unlabeled, e.g., in a zero-shot scenario, the direct evaluation of the model on the target corpus becomes unfeasible. Unsupervised selection of the most effective pre-trained dense retriever becomes then a crucial challenge. Current methods for dense retriever selection are insufficient in handling scenarios with domain shift. Our proposed solution leverages LLMs to generate pseudo-relevant queries, labels and reference lists based on a set of documents sampled from the target corpus. Dense retrievers are then ranked based on their effectiveness on these generated pseudo-relevant signals. Notably, our method is the first approach that relies solely on the target corpus, eliminating the need for both training corpora and test labels. To evaluate the effectiveness of our method, we construct a large pool of state-of-the-art dense retrievers. The proposed approach outperforms existing baselines with respect to both dense retriever selection and ranking. We make our code and results publicly available at https://github.com/ielab/larmor/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15818v1">DuanzAI: Slang-Enhanced LLM with Prompt for Humor Understanding</a></div>
    <div class="paper-meta">
      📅 2024-05-23
    </div>
    <details class="paper-abstract">
      Language's complexity is evident in the rich tapestry of slang expressions, often laden with humor and cultural nuances. This linguistic phenomenon has become increasingly prevalent, especially in digital communication. However, existing AI models, including ChatGPT-3.5, face challenges in comprehending these nuances, particularly in Chinese slang. In this study, we present DuanzAI, an innovative approach enhancing Large Language Models (LLMs) with deep Chinese slang comprehension. Leveraging curated datasets and advanced techniques, DuanzAI bridges the gap between human expression and AI comprehension, enabling contextually relevant responses. Our experiments contrast LLMs' performance with a custom Punchline Entity Recognition (PER) system, integrating phonetic matching and pinyin2hanzi techniques. Applying these insights, we developed ChatDAI, an advanced chatbot and released our code at \url{https://github.com/YesianRohn/DuanzAI}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14023v1">WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response</a></div>
    <div class="paper-meta">
      📅 2024-05-22
    </div>
    <details class="paper-abstract">
      The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13954v1">What is Your Data Worth to GPT? LLM-Scale Data Valuation with Influence Functions</a></div>
    <div class="paper-meta">
      📅 2024-05-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on a vast amount of human-written data, but data providers often remain uncredited. In response to this issue, data valuation (or data attribution), which quantifies the contribution or value of each data to the model output, has been discussed as a potential solution. Nevertheless, applying existing data valuation methods to recent LLMs and their vast training datasets has been largely limited by prohibitive compute and memory costs. In this work, we focus on influence functions, a popular gradient-based data valuation method, and significantly improve its scalability with an efficient gradient projection strategy called LoGra that leverages the gradient structure in backpropagation. We then provide a theoretical motivation of gradient projection approaches to influence functions to promote trust in the data valuation process. Lastly, we lower the barrier to implementing data valuation systems by introducing LogIX, a software package that can transform existing training code into data valuation code with minimal effort. In our data valuation experiments, LoGra achieves competitive accuracy against more expensive baselines while showing up to 6,500x improvement in throughput and 5x reduction in GPU memory usage when applied to Llama3-8B-Instruct and the 1B-token dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13949v1">PitVQA: Image-grounded Text Embedding LLM for Visual Question Answering in Pituitary Surgery</a></div>
    <div class="paper-meta">
      📅 2024-05-22
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Visual Question Answering (VQA) within the surgical domain, utilizing Large Language Models (LLMs), offers a distinct opportunity to improve intra-operative decision-making and facilitate intuitive surgeon-AI interaction. However, the development of LLMs for surgical VQA is hindered by the scarcity of diverse and extensive datasets with complex reasoning tasks. Moreover, contextual fusion of the image and text modalities remains an open research challenge due to the inherent differences between these two types of information and the complexity involved in aligning them. This paper introduces PitVQA, a novel dataset specifically designed for VQA in endonasal pituitary surgery and PitVQA-Net, an adaptation of the GPT2 with a novel image-grounded text embedding for surgical VQA. PitVQA comprises 25 procedural videos and a rich collection of question-answer pairs spanning crucial surgical aspects such as phase and step recognition, context understanding, tool detection and localization, and tool-tissue interactions. PitVQA-Net consists of a novel image-grounded text embedding that projects image and text features into a shared embedding space and GPT2 Backbone with an excitation block classification head to generate contextually relevant answers within the complex domain of endonasal pituitary surgery. Our image-grounded text embedding leverages joint embedding, cross-attention and contextual representation to understand the contextual relationship between questions and surgical images. We demonstrate the effectiveness of PitVQA-Net on both the PitVQA and the publicly available EndoVis18-VQA dataset, achieving improvements in balanced accuracy of 8% and 9% over the most recent baselines, respectively. Our code and dataset is available at https://github.com/mobarakol/PitVQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13932v1">Chain of Targeted Verification Questions to Improve the Reliability of Code Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-22
      | 💬 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLM-based assistants, such as GitHub Copilot and ChatGPT, have the potential to generate code that fulfills a programming task described in a natural language description, referred to as a prompt. The widespread accessibility of these assistants enables users with diverse backgrounds to generate code and integrate it into software projects. However, studies show that code generated by LLMs is prone to bugs and may miss various corner cases in task specifications. Presenting such buggy code to users can impact their reliability and trust in LLM-based assistants. Moreover, significant efforts are required by the user to detect and repair any bug present in the code, especially if no test cases are available. In this study, we propose a self-refinement method aimed at improving the reliability of code generated by LLMs by minimizing the number of bugs before execution, without human intervention, and in the absence of test cases. Our approach is based on targeted Verification Questions (VQs) to identify potential bugs within the initial code. These VQs target various nodes within the Abstract Syntax Tree (AST) of the initial code, which have the potential to trigger specific types of bug patterns commonly found in LLM-generated code. Finally, our method attempts to repair these potential bugs by re-prompting the LLM with the targeted VQs and the initial code. Our evaluation, based on programming tasks in the CoderEval dataset, demonstrates that our proposed method outperforms state-of-the-art methods by decreasing the number of targeted errors in the code between 21% to 62% and improving the number of executable code instances to 13%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03066v2">A scoping review of using Large Language Models (LLMs) to investigate Electronic Health Records (EHRs)</a></div>
    <div class="paper-meta">
      📅 2024-05-22
    </div>
    <details class="paper-abstract">
      Electronic Health Records (EHRs) play an important role in the healthcare system. However, their complexity and vast volume pose significant challenges to data interpretation and analysis. Recent advancements in Artificial Intelligence (AI), particularly the development of Large Language Models (LLMs), open up new opportunities for researchers in this domain. Although prior studies have demonstrated their potential in language understanding and processing in the context of EHRs, a comprehensive scoping review is lacking. This study aims to bridge this research gap by conducting a scoping review based on 329 related papers collected from OpenAlex. We first performed a bibliometric analysis to examine paper trends, model applications, and collaboration networks. Next, we manually reviewed and categorized each paper into one of the seven identified topics: named entity recognition, information extraction, text similarity, text summarization, text classification, dialogue system, and diagnosis and prediction. For each topic, we discussed the unique capabilities of LLMs, such as their ability to understand context, capture semantic relations, and generate human-like text. Finally, we highlighted several implications for researchers from the perspectives of data resources, prompt engineering, fine-tuning, performance measures, and ethical concerns. In conclusion, this study provides valuable insights into the potential of LLMs to transform EHR research and discusses their applications and ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15406v2">Wiki-LLaVA: Hierarchical Retrieval-Augmented Generation for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-22
      | 💬 CVPR 2024 Workshop on What is Next in Multimodal Foundation Models
    </div>
    <details class="paper-abstract">
      Multimodal LLMs are the natural evolution of LLMs, and enlarge their capabilities so as to work beyond the pure textual modality. As research is being carried out to design novel architectures and vision-and-language adapters, in this paper we concentrate on endowing such models with the capability of answering questions that require external knowledge. Our approach, termed Wiki-LLaVA, aims at integrating an external knowledge source of multimodal documents, which is accessed through a hierarchical retrieval pipeline. Relevant passages, using this approach, are retrieved from the external knowledge source and employed as additional context for the LLM, augmenting the effectiveness and precision of generated dialogues. We conduct extensive experiments on datasets tailored for visual question answering with external data and demonstrate the appropriateness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06899v6">Flames: Benchmarking Value Alignment of LLMs in Chinese</a></div>
    <div class="paper-meta">
      📅 2024-05-22
      | 💬 Accepted to the NAACL 2024
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) across various regions underscores the urgent need to evaluate their alignment with human values. Current benchmarks, however, fall short of effectively uncovering safety vulnerabilities in LLMs. Despite numerous models achieving high scores and 'topping the chart' in these evaluations, there is still a significant gap in LLMs' deeper alignment with human values and achieving genuine harmlessness. To this end, this paper proposes a value alignment benchmark named Flames, which encompasses both common harmlessness principles and a unique morality dimension that integrates specific Chinese values such as harmony. Accordingly, we carefully design adversarial prompts that incorporate complex scenarios and jailbreaking methods, mostly with implicit malice. By prompting 17 mainstream LLMs, we obtain model responses and rigorously annotate them for detailed evaluation. Our findings indicate that all the evaluated LLMs demonstrate relatively poor performance on Flames, particularly in the safety and fairness dimensions. We also develop a lightweight specified scorer capable of scoring LLMs across multiple dimensions to efficiently evaluate new models on the benchmark. The complexity of Flames has far exceeded existing benchmarks, setting a new challenge for contemporary LLMs and highlighting the need for further alignment of LLMs. Our benchmark is publicly available at https://github.com/AIFlames/Flames.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13358v1">AdpQ: A Zero-shot Calibration Free Adaptive Post Training Quantization Method for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-22
    </div>
    <details class="paper-abstract">
      The ever-growing computational complexity of Large Language Models (LLMs) necessitates efficient deployment strategies. The current state-of-the-art approaches for Post-training Quantization (PTQ) often require calibration to achieve the desired accuracy. This paper presents AdpQ, a novel zero-shot adaptive PTQ method for LLMs that achieves the state-of-the-art performance in low-precision quantization (e.g. 3-bit) without requiring any calibration data. Inspired by Adaptive LASSO regression model, our proposed approach tackles the challenge of outlier activations by separating salient weights using an adaptive soft-thresholding method. Guided by Adaptive LASSO, this method ensures that the quantized weights distribution closely follows the originally trained weights and eliminates the need for calibration data entirely, setting our method apart from popular approaches such as SpQR and AWQ. Furthermore, our method offers an additional benefit in terms of privacy preservation by eliminating any calibration or training data. We also delve deeper into the information-theoretic underpinnings of the proposed method. We demonstrate that it leverages the Adaptive LASSO to minimize the Kullback-Leibler divergence between the quantized weights and the originally trained weights. This minimization ensures the quantized model retains the Shannon information content of the original model to a great extent, guaranteeing efficient deployment without sacrificing accuracy or information. Our results achieve the same accuracy as the existing methods on various LLM benchmarks while the quantization time is reduced by at least 10x, solidifying our contribution to efficient and privacy-preserving LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06907v2">AIOS Compiler: LLM as Interpreter for Natural Language Programming and Flow Programming of AI Agents</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 12 pages, 6 figures, comments and suggestions are welcome
    </div>
    <details class="paper-abstract">
      Since their inception, programming languages have trended towards greater readability and lower barriers for programmers. Following this trend, natural language can be a promising type of programming language that provides great flexibility and usability and helps towards the democracy of programming. However, the inherent vagueness, ambiguity, and verbosity of natural language pose significant challenges in developing an interpreter that can accurately understand the programming logic and execute instructions written in natural language. Fortunately, recent advancements in Large Language Models (LLMs) have demonstrated remarkable proficiency in interpreting complex natural language. Inspired by this, we develop a novel system for Code Representation and Execution (CoRE), which employs LLM as interpreter to interpret and execute natural language instructions. The proposed system unifies natural language programming, pseudo-code programming, and flow programming under the same representation for constructing language agents, while LLM serves as the interpreter to interpret and execute the agent programs. In this paper, we begin with defining the programming syntax that structures natural language instructions logically. During the execution, we incorporate external memory to minimize redundancy. Furthermore, we equip the designed interpreter with the capability to invoke external tools, compensating for the limitations of LLM in specialized domains or when accessing real-time information. This work is open-source at https://github.com/agiresearch/CoRE, https://github.com/agiresearch/OpenAGI, and https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13181v1">Comparative Analysis of Different Efficient Fine Tuning Methods of Large Language Models (LLMs) in Low-Resource Setting</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 9 pages of main paper, 1 page of references, 6 appendix pages, 11 figures, 18 tables
    </div>
    <details class="paper-abstract">
      In the domain of large language models (LLMs), arXiv:2305.16938 showed that few-shot full-model fine-tuning -- namely Vanilla Fine Tuning (FT) and Pattern-Based Fine Tuning (PBFT) --, and In-Context Learning (ICL) generalize similarly on Out-Of-Domain (OOD) datasets, but vary in terms of task adaptation. However, they both pose challenges, especially in term of memory requirements. In this paper, we further try to push the understanding of different fine-tuning strategies for LLM and aim to bring a myriad of these on the same pedestal for an elaborate comparison with full-model fine-tuning on two diverse datasets. To that end, we conducted a series of experiments, beginning with state-of-the-art methods like vanilla fine-tuning and Pattern-Based Fine-Tuning (PBFT) on pre-trained models across two datasets, COLA and MNLI. We then investigate adaptive fine-tuning and the efficiency of LoRA adapters in a few-shot setting. Finally, we also compare an alternative approach that has gained recent popularity -- context distillation -- with the vanilla FT and PBFT with and without few-shot setup. Our findings suggest that these alternative strategies that we explored can exhibit out-of-domain generalization comparable to that of vanilla FT and PBFT. PBFT under-performs Vanilla FT on out-of-domain (OOD) data, emphasizing the need for effective prompts. Further, our adaptive-fine tuning and LoRA experiments perform comparable or slightly worse than the standard fine-tunings as anticipated, since standard fine-tunings involve tuning the entire model. Finally, our context distillation experiments out-perform the standard fine-tuning methods. These findings underscore that eventually the choice of an appropriate fine-tuning method depends on the available resources (memory, compute, data) and task adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12946v1">Tutorly: Turning Programming Videos Into Apprenticeship Learning Environments with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      Online programming videos, including tutorials and streamcasts, are widely popular and contain a wealth of expert knowledge. However, effectively utilizing these resources to achieve targeted learning goals can be challenging. Unlike direct tutoring, video content lacks tailored guidance based on individual learning paces, personalized feedback, and interactive engagement necessary for support and monitoring. Our work transforms programming videos into one-on-one tutoring experiences using the cognitive apprenticeship framework. Tutorly, developed as a JupyterLab Plugin, allows learners to (1) set personalized learning goals, (2) engage in learning-by-doing through a conversational LLM-based mentor agent, (3) receive guidance and feedback based on a student model that steers the mentor moves. In a within-subject study with 16 participants learning exploratory data analysis from a streamcast, Tutorly significantly improved their performance from 61.9% to 76.6% based on a post-test questionnaire. Tutorly demonstrates the potential for enhancing programming video learning experiences with LLM and learner modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12842v1">SmartFlow: Robotic Process Automation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 32nd ACM International Conference on Information and Knowledge Management
    </div>
    <details class="paper-abstract">
      Robotic Process Automation (RPA) systems face challenges in handling complex processes and diverse screen layouts that require advanced human-like decision-making capabilities. These systems typically rely on pixel-level encoding through drag-and-drop or automation frameworks such as Selenium to create navigation workflows, rather than visual understanding of screen elements. In this context, we present SmartFlow, an AI-based RPA system that uses pre-trained large language models (LLMs) coupled with deep-learning based image understanding. Our system can adapt to new scenarios, including changes in the user interface and variations in input data, without the need for human intervention. SmartFlow uses computer vision and natural language processing to perceive visible elements on the graphical user interface (GUI) and convert them into a textual representation. This information is then utilized by LLMs to generate a sequence of actions that are executed by a scripting engine to complete an assigned task. To assess the effectiveness of SmartFlow, we have developed a dataset that includes a set of generic enterprise applications with diverse layouts, which we are releasing for research use. Our evaluations on this dataset demonstrate that SmartFlow exhibits robustness across different layouts and applications. SmartFlow can automate a wide range of business processes such as form filling, customer service, invoice processing, and back-office operations. SmartFlow can thus assist organizations in enhancing productivity by automating an even larger fraction of screen-based workflows. The demo-video and dataset are available at https://smartflow-4c5a0a.webflow.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13095v1">Presentations are not always linear! GNN meets LLM for Document-to-Presentation Transformation with Attribution</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 This paper is under review in a conference
    </div>
    <details class="paper-abstract">
      Automatically generating a presentation from the text of a long document is a challenging and useful problem. In contrast to a flat summary, a presentation needs to have a better and non-linear narrative, i.e., the content of a slide can come from different and non-contiguous parts of the given document. However, it is difficult to incorporate such non-linear mapping of content to slides and ensure that the content is faithful to the document. LLMs are prone to hallucination and their performance degrades with the length of the input document. Towards this, we propose a novel graph based solution where we learn a graph from the input document and use a combination of graph neural network and LLM to generate a presentation with attribution of content for each slide. We conduct thorough experiments to show the merit of our approach compared to directly using LLMs for this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11514v2">Towards Translating Real-World Code with LLMs: A Study of Translating to Rust</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 11 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in code translation - the task of translating code written in one programming language to another language - due to their ability to write code in most programming languages. However, LLM's effectiveness on translating real-world code remains largely unstudied. In this work, we perform the first substantial study on LLM-based translation to Rust by assessing the ability of five state-of-the-art LLMs, GPT4, Claude 3, Claude 2.1, Gemini Pro, and Mixtral. We conduct our study on code extracted from real-world open source projects. To enable our study, we develop FLOURINE, an end-to-end code translation tool that uses differential fuzzing to check if a Rust translation is I/O equivalent to the original source program, eliminating the need for pre-existing test cases. As part of our investigation, we assess both the LLM's ability to produce an initially successful translation, as well as their capacity to fix a previously generated buggy one. If the original and the translated programs are not I/O equivalent, we apply a set of automated feedback strategies, including feedback to the LLM with counterexamples. Our results show that the most successful LLM can translate 47% of our benchmarks, and also provides insights into next steps for improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12766v1">Test Oracle Automation in the era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      The effectiveness of a test suite in detecting faults highly depends on the correctness and completeness of its test oracles. Large Language Models (LLMs) have already demonstrated remarkable proficiency in tackling diverse software testing tasks, such as automated test generation and program repair. This paper aims to enable discussions on the potential of using LLMs for test oracle automation, along with the challenges that may emerge during the generation of various types of oracles. Additionally, our aim is to initiate discussions on the primary threats that SE researchers must consider when employing LLMs for oracle automation, encompassing concerns regarding oracle deficiencies and data leakages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12541v1">DrHouse: An LLM-empowered Diagnostic Reasoning System through Harnessing Outcomes from Sensor Data and Expert Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have the potential to transform digital healthcare, as evidenced by recent advances in LLM-based virtual doctors. However, current approaches rely on patient's subjective descriptions of symptoms, causing increased misdiagnosis. Recognizing the value of daily data from smart devices, we introduce a novel LLM-based multi-turn consultation virtual doctor system, DrHouse, which incorporates three significant contributions: 1) It utilizes sensor data from smart devices in the diagnosis process, enhancing accuracy and reliability. 2) DrHouse leverages continuously updating medical databases such as Up-to-Date and PubMed to ensure our model remains at diagnostic standard's forefront. 3) DrHouse introduces a novel diagnostic algorithm that concurrently evaluates potential diseases and their likelihood, facilitating more nuanced and informed medical assessments. Through multi-turn interactions, DrHouse determines the next steps, such as accessing daily data from smart devices or requesting in-lab tests, and progressively refines its diagnoses. Evaluations on three public datasets and our self-collected datasets show that DrHouse can achieve up to an 18.8% increase in diagnosis accuracy over the state-of-the-art baselines. The results of a 32-participant user study show that 75% medical experts and 91.7% patients are willing to use DrHouse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12528v1">SirLLM: Streaming Infinite Retentive LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent in various domains, their ability to process inputs of any length and maintain a degree of memory becomes essential. However, the one-off input of overly long texts is limited, as studies have shown that when input lengths exceed the LLMs' pre-trained text length, there is a dramatic decline in text generation capabilities. Moreover, simply extending the length of pre-training texts is impractical due to the difficulty in obtaining long text data and the substantial memory consumption costs this would entail for LLMs. Recent efforts have employed streaming inputs to alleviate the pressure of excessively long text inputs, but this approach can significantly impair the model's long-term memory capabilities. Motivated by this challenge, we introduce Streaming Infinite Retentive LLM (SirLLM), which allows LLMs to maintain longer memory during infinite-length dialogues without the need for fine-tuning. SirLLM utilizes the Token Entropy metric and a memory decay mechanism to filter key phrases, endowing LLMs with both long-lasting and flexible memory. We designed three distinct tasks and constructed three datasets to measure the effectiveness of SirLLM from various angles: (1) DailyDialog; (2) Grocery Shopping; (3) Rock-Paper-Scissors. Our experimental results robustly demonstrate that SirLLM can achieve stable and significant improvements across different LLMs and tasks, compellingly proving its effectiveness. When having a coversation, "A sir could forget himself," but SirLLM never does! Our code is publicly available at https://github.com/Zoeyyao27/SirLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06713v2">Unveiling the Competitive Dynamics: A Comparative Evaluation of American and Chinese LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-21
      | 💬 There was a miscommunication among the co-authors, resulting in the accidental submission of this paper to arXiv. We are in need of withdrawing the paper from your platform
    </div>
    <details class="paper-abstract">
      The strategic significance of Large Language Models (LLMs) in economic expansion, innovation, societal development, and national security has been increasingly recognized since the advent of ChatGPT. This study provides a comprehensive comparative evaluation of American and Chinese LLMs in both English and Chinese contexts. We proposed a comprehensive evaluation framework that encompasses natural language proficiency, disciplinary expertise, and safety and responsibility, and systematically assessed 16 prominent models from the US and China under various operational tasks and scenarios. Our key findings show that GPT 4-Turbo is at the forefront in English contexts, whereas Ernie-Bot 4 stands out in Chinese contexts. The study also highlights disparities in LLM performance across languages and tasks, stressing the necessity for linguistically and culturally nuanced model development. The complementary strengths of American and Chinese LLMs point to the value of Sino-US collaboration in advancing LLM technology. The research presents the current LLM competition landscape and offers valuable insights for policymakers and businesses regarding strategic LLM investments and development. Future work will expand on this framework to include emerging LLM multimodal capabilities and business application assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05465v2">Vidur: A Large-Scale Simulation Framework For LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      Optimizing the deployment of Large language models (LLMs) is expensive today since it requires experimentally running an application workload against an LLM implementation while exploring large configuration space formed by system knobs such as parallelization strategies, batching techniques, and scheduling policies. To address this challenge, we present Vidur - a large-scale, high-fidelity, easily-extensible simulation framework for LLM inference performance. Vidur models the performance of LLM operators using a combination of experimental profiling and predictive modeling, and evaluates the end-to-end inference performance for different workloads by estimating several metrics of interest such as latency and throughput. We validate the fidelity of Vidur on several LLMs and show that it estimates inference latency with less than 9% error across the range. Further, we present Vidur-Search, a configuration search tool that helps optimize LLM deployment. Vidur-Search uses Vidur to automatically identify the most cost-effective deployment configuration that meets application performance constraints. For example, Vidur-Search finds the best deployment configuration for LLaMA2-70B in one hour on a CPU machine, in contrast to a deployment-based exploration which would require 42K GPU hours - costing ~218K dollars. Source code for Vidur is available at https://github.com/microsoft/vidur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11613v2">Decoding by Contrasting Knowledge: Enhancing LLMs' Confidence on Edited Facts</a></div>
    <div class="paper-meta">
      📅 2024-05-21
    </div>
    <details class="paper-abstract">
      The knowledge within large language models (LLMs) may become outdated quickly. While in-context editing (ICE) is currently the most effective method for knowledge editing (KE), it is constrained by the black-box modeling of LLMs and thus lacks interpretability. Our work aims to elucidate the superior performance of ICE on the KE by analyzing the impacts of in-context new knowledge on token-wise distributions. We observe that despite a significant boost in logits of the new knowledge, the performance of is still hindered by stubborn knowledge. Stubborn knowledge refers to as facts that have gained excessive confidence during pretraining, making it hard to edit effectively. To address this issue and further enhance the performance of ICE, we propose a novel approach termed $\textbf{De}$coding by $\textbf{C}$ontrasting $\textbf{K}$nowledge (DeCK). DeCK derives the distribution of the next token by contrasting the logits obtained from the newly edited knowledge guided by ICE with those from the unedited parametric knowledge. Our experiments consistently demonstrate that DeCK enhances the confidence of LLMs in edited facts. For instance, it improves the performance of LLaMA3-8B-instruct on MQuAKE by up to 219%, demonstrating its capability to strengthen ICE in the editing of stubborn knowledge. Our work paves the way to develop the both effective and accountable KE methods for LLMs. (The source code is available at: https://deck-llm.meirtz.com)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.18028v2">Evaluating the Capabilities of LLMs for Supporting Anticipatory Impact Assessment</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 10 pages + research ethics and social impact statement, references, and appendix. Under conference review
    </div>
    <details class="paper-abstract">
      Gaining insight into the potential negative impacts of emerging Artificial Intelligence (AI) technologies in society is a challenge for implementing anticipatory governance approaches. One approach to produce such insight is to use Large Language Models (LLMs) to support and guide experts in the process of ideating and exploring the range of undesirable consequences of emerging technologies. However, performance evaluations of LLMs for such tasks are still needed, including examining the general quality of generated impacts but also the range of types of impacts produced and resulting biases. In this paper, we demonstrate the potential for generating high-quality and diverse impacts of AI in society by fine-tuning completion models (GPT-3 and Mistral-7B) on a diverse sample of articles from news media and comparing those outputs to the impacts generated by instruction-based (GPT-4 and Mistral-7B-Instruct) models. We examine the generated impacts for coherence, structure, relevance, and plausibility and find that the generated impacts using Mistral-7B, a small open-source model fine-tuned on impacts from the news media, tend to be qualitatively on par with impacts generated using a more capable and larger scale model such as GPT-4. Moreover, we find that impacts produced by instruction-based models had gaps in the production of certain categories of impacts in comparison to fine-tuned models. This research highlights a potential bias in the range of impacts generated by state-of-the-art LLMs and the potential of aligning smaller LLMs on news media as a scalable alternative to generate high quality and more diverse impacts in support of anticipatory governance approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12347v1">Self-HWDebug: Automation of LLM Self-Instructing for Hardware Security Verification</a></div>
    <div class="paper-meta">
      📅 2024-05-20
    </div>
    <details class="paper-abstract">
      The rise of instruction-tuned Large Language Models (LLMs) marks a significant advancement in artificial intelligence (AI) (tailored to respond to specific prompts). Despite their popularity, applying such models to debug security vulnerabilities in hardware designs, i.e., register transfer language (RTL) modules, particularly at system-on-chip (SoC) level, presents considerable challenges. One of the main issues lies in the need for precisely designed instructions for pinpointing and mitigating the vulnerabilities, which requires substantial time and expertise from human experts. In response to this challenge, this paper proposes Self-HWDebug, an innovative framework that leverages LLMs to automatically create required debugging instructions. In Self-HWDebug, a set of already identified bugs from the most critical hardware common weakness enumeration (CWE) listings, along with mitigation resolutions, is provided to the framework, followed by prompting the LLMs to generate targeted instructions for such mitigation. The LLM-generated instructions are subsequently used as references to address vulnerabilities within the same CWE category but in totally different designs, effectively demonstrating the framework's ability to extend solutions across related security issues. Self-HWDebug significantly reduces human intervention by using the model's own output to guide debugging. Through comprehensive testing, Self-HWDebug proves not only to reduce experts' effort/time but also to even improve the quality of the debugging process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12209v1">MathBench: Evaluating the Theory and Application Proficiency of LLMs with a Hierarchical Mathematics Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 Project: https://github.com/open-compass/MathBench
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have showcased significant improvements in mathematics. However, traditional math benchmarks like GSM8k offer a unidimensional perspective, falling short in providing a holistic assessment of the LLMs' math capabilities. To address this gap, we introduce MathBench, a new benchmark that rigorously assesses the mathematical capabilities of large language models. MathBench spans a wide range of mathematical disciplines, offering a detailed evaluation of both theoretical understanding and practical problem-solving skills. The benchmark progresses through five distinct stages, from basic arithmetic to college mathematics, and is structured to evaluate models at various depths of knowledge. Each stage includes theoretical questions and application problems, allowing us to measure a model's mathematical proficiency and its ability to apply concepts in practical scenarios. MathBench aims to enhance the evaluation of LLMs' mathematical abilities, providing a nuanced view of their knowledge understanding levels and problem solving skills in a bilingual context. The project is released at https://github.com/open-compass/MathBench .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12205v1">Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Metacognitive knowledge refers to humans' intuitive knowledge of their own thinking and reasoning processes. Today's best LLMs clearly possess some reasoning processes. The paper gives evidence that they also have metacognitive knowledge, including ability to name skills and procedures to apply given a task. We explore this primarily in context of math reasoning, developing a prompt-guided interaction procedure to get a powerful LLM to assign sensible skill labels to math questions, followed by having it perform semantic clustering to obtain coarser families of skill labels. These coarse skill labels look interpretable to humans. To validate that these skill labels are meaningful and relevant to the LLM's reasoning processes we perform the following experiments. (a) We ask GPT-4 to assign skill labels to training questions in math datasets GSM8K and MATH. (b) When using an LLM to solve the test questions, we present it with the full list of skill labels and ask it to identify the skill needed. Then it is presented with randomly selected exemplar solved questions associated with that skill label. This improves accuracy on GSM8k and MATH for several strong LLMs, including code-assisted models. The methodology presented is domain-agnostic, even though this article applies it to math problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18677v2">Splitwise: Efficient generative LLM inference using phase splitting</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 12 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Recent innovations in generative large language models (LLMs) have made their applications and use-cases ubiquitous. This has led to large-scale deployments of these models, using complex, expensive, and power-hungry AI accelerators, most commonly GPUs. These developments make LLM inference efficiency an important challenge. Based on our extensive characterization, we find that there are two main phases during an LLM inference request: a compute-intensive prompt computation, and a memory-intensive token generation, each with distinct latency, throughput, memory, and power characteristics. Despite state-of-the-art batching and scheduling, the token generation phase underutilizes compute resources. Specifically, unlike compute-intensive prompt computation phases, token generation phases do not require the compute capability of the latest GPUs, and can be run with lower power and cost. With Splitwise, we propose splitting the two phases of a LLM inference request on to separate machines. This allows us to use hardware that is well-suited for each phase, and provision resources independently per phase. However, splitting an inference request across machines requires state transfer from the machine running prompt computation over to the machine generating tokens. We implement and optimize this state transfer using the fast back-plane interconnects available in today's GPU clusters. We use the Splitwise technique to design LLM inference clusters using the same or different types of machines for the prompt computation and token generation phases. Our clusters are optimized for three key objectives: throughput, cost, and power. In particular, we show that we can achieve 1.4x higher throughput at 20% lower cost than current designs. Alternatively, we can achieve 2.35x more throughput with the same cost and power budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15491v2">API-BLEND: A Comprehensive Corpora for Training and Benchmarking API LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 Accepted at ACL'24-main conference
    </div>
    <details class="paper-abstract">
      There is a growing need for Large Language Models (LLMs) to effectively use tools and external Application Programming Interfaces (APIs) to plan and complete tasks. As such, there is tremendous interest in methods that can acquire sufficient quantities of train and test data that involve calls to tools / APIs. Two lines of research have emerged as the predominant strategies for addressing this challenge. The first has focused on synthetic data generation techniques, while the second has involved curating task-adjacent datasets which can be transformed into API / Tool-based tasks. In this paper, we focus on the task of identifying, curating, and transforming existing datasets and, in turn, introduce API-BLEND, a large corpora for training and systematic testing of tool-augmented LLMs. The datasets mimic real-world scenarios involving API-tasks such as API / tool detection, slot filling, and sequencing of the detected APIs. We demonstrate the utility of the API-BLEND dataset for both training and benchmarking purposes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11880v1">Quantifying In-Context Reasoning Effects and Memorization Effects in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-20
    </div>
    <details class="paper-abstract">
      In this study, we propose an axiomatic system to define and quantify the precise memorization and in-context reasoning effects used by the large language model (LLM) for language generation. These effects are formulated as non-linear interactions between tokens/words encoded by the LLM. Specifically, the axiomatic system enables us to categorize the memorization effects into foundational memorization effects and chaotic memorization effects, and further classify in-context reasoning effects into enhanced inference patterns, eliminated inference patterns, and reversed inference patterns. Besides, the decomposed effects satisfy the sparsity property and the universal matching property, which mathematically guarantee that the LLM's confidence score can be faithfully decomposed into the memorization effects and in-context reasoning effects. Experiments show that the clear disentanglement of memorization effects and in-context reasoning effects enables a straightforward examination of detailed inference patterns encoded by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10877v2">Incubating Text Classifiers Following User Instruction with Nothing but LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-20
    </div>
    <details class="paper-abstract">
      In this paper, we aim to generate text classification data given arbitrary class definitions (i.e., user instruction), so one can train a small text classifier without any human annotation or raw corpus. Compared with pioneer attempts, our proposed Incubator is the first framework that can handle complicated and even mutually dependent classes (e.g., "TED Talk given by Educator" and "Other"). Specifically, Incubator is an LLM firstly tuned on the instruction-to-data mappings that we obtained from classification datasets and descriptions on HuggingFace together with in-context augmentation by GPT-4. We then refine Incubator by learning on the cluster centers of semantic textual embeddings to emphasize the uniformity and semantic diversity in generations. We compare Incubator on various classification tasks with strong baselines such as direct LLM-based inference and training data generation by prompt engineering. Experiments show Incubator is able to (1) perform well on traditional benchmarks, (2) take label dependency and user preference into consideration, and (3) enable logical text mining by incubating multiple classifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11706v1">Increasing the LLM Accuracy for Question Answering: Ontologies to the Rescue!</a></div>
    <div class="paper-meta">
      📅 2024-05-20
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      There is increasing evidence that question-answering (QA) systems with Large Language Models (LLMs), which employ a knowledge graph/semantic representation of an enterprise SQL database (i.e. Text-to-SPARQL), achieve higher accuracy compared to systems that answer questions directly on SQL databases (i.e. Text-to-SQL). Our previous benchmark research showed that by using a knowledge graph, the accuracy improved from 16% to 54%. The question remains: how can we further improve the accuracy and reduce the error rate? Building on the observations of our previous research where the inaccurate LLM-generated SPARQL queries followed incorrect paths, we present an approach that consists of 1) Ontology-based Query Check (OBQC): detects errors by leveraging the ontology of the knowledge graph to check if the LLM-generated SPARQL query matches the semantic of ontology and 2) LLM Repair: use the error explanations with an LLM to repair the SPARQL query. Using the chat with the data benchmark, our primary finding is that our approach increases the overall accuracy to 72% including an additional 8% of "I don't know" unknown results. Thus, the overall error rate is 20%. These results provide further evidence that investing knowledge graphs, namely the ontology, provides higher accuracy for LLM powered question answering systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11591v1">Generative Students: Using LLM-Simulated Student Profiles to Support Question Item Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-05-19
      | 💬 To be published in L@S'24: Proceedings of the Eleventh ACM Conference on Learning @ Scale
    </div>
    <details class="paper-abstract">
      Evaluating the quality of automatically generated question items has been a long standing challenge. In this paper, we leverage LLMs to simulate student profiles and generate responses to multiple-choice questions (MCQs). The generative students' responses to MCQs can further support question item evaluation. We propose Generative Students, a prompt architecture designed based on the KLI framework. A generative student profile is a function of the list of knowledge components the student has mastered, has confusion about or has no evidence of knowledge of. We instantiate the Generative Students concept on the subject domain of heuristic evaluation. We created 45 generative students using GPT-4 and had them respond to 20 MCQs. We found that the generative students produced logical and believable responses that were aligned with their profiles. We then compared the generative students' responses to real students' responses on the same set of MCQs and found a high correlation. Moreover, there was considerable overlap in the difficult questions identified by generative students and real students. A subsequent case study demonstrated that an instructor could improve question quality based on the signals provided by Generative Students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11446v1">MAML-en-LLM: Model Agnostic Meta-Training of LLMs for Improved In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2024-05-19
      | 💬 KDD 2024, 11 pages(9 main, 2 ref, 1 App) Openreview https://openreview.net/forum?id=JwecLNhWDy&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DKDD.org%2F2024%2FResearch_Track%2FAuthors%23your-submissions)
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to unseen tasks with in-context training samples without fine-tuning remains an important research problem. To learn a robust LLM that adapts well to unseen tasks, multiple meta-training approaches have been proposed such as MetaICL and MetaICT, which involve meta-training pre-trained LLMs on a wide variety of diverse tasks. These meta-training approaches essentially perform in-context multi-task fine-tuning and evaluate on a disjointed test set of tasks. Even though they achieve impressive performance, their goal is never to compute a truly general set of parameters. In this paper, we propose MAML-en-LLM, a novel method for meta-training LLMs, which can learn truly generalizable parameters that not only perform well on disjointed tasks but also adapts to unseen tasks. We see an average increase of 2% on unseen domains in the performance while a massive 4% improvement on adaptation performance. Furthermore, we demonstrate that MAML-en-LLM outperforms baselines in settings with limited amount of training data on both seen and unseen domains by an average of 2%. Finally, we discuss the effects of type of tasks, optimizers and task complexity, an avenue barely explored in meta-training literature. Exhaustive experiments across 7 task settings along with two data settings demonstrate that models trained with MAML-en-LLM outperform SOTA meta-training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.14274v4">Multi-role Consensus through LLMs Discussions for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2024-05-18
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have highlighted the potential for vulnerability detection, a crucial component of software quality assurance. Despite this progress, most studies have been limited to the perspective of a single role, usually testers, lacking diverse viewpoints from different roles in a typical software development life-cycle, including both developers and testers. To this end, this paper introduces a multi-role approach to employ LLMs to act as different roles simulating a real-life code review process and engaging in discussions toward a consensus on the existence and classification of vulnerabilities in the code. Preliminary evaluation of this approach indicates a 13.48% increase in the precision rate, an 18.25% increase in the recall rate, and a 16.13% increase in the F1 score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11273v1">Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts</a></div>
    <div class="paper-meta">
      📅 2024-05-18
      | 💬 22 pages, 13 figures. Project Website: https://uni-moe.github.io/. Working in progress
    </div>
    <details class="paper-abstract">
      Recent advancements in Multimodal Large Language Models (MLLMs) underscore the significance of scalable models and data to boost performance, yet this often incurs substantial computational costs. Although the Mixture of Experts (MoE) architecture has been employed to efficiently scale large language and image-text models, these efforts typically involve fewer experts and limited modalities. To address this, our work presents the pioneering attempt to develop a unified MLLM with the MoE architecture, named Uni-MoE that can handle a wide array of modalities. Specifically, it features modality-specific encoders with connectors for a unified multimodal representation. We also implement a sparse MoE architecture within the LLMs to enable efficient training and inference through modality-level data parallelism and expert-level model parallelism. To enhance the multi-expert collaboration and generalization, we present a progressive training strategy: 1) Cross-modality alignment using various connectors with different cross-modality data, 2) Training modality-specific experts with cross-modality instruction data to activate experts' preferences, and 3) Tuning the Uni-MoE framework utilizing Low-Rank Adaptation (LoRA) on mixed multimodal instruction data. We evaluate the instruction-tuned Uni-MoE on a comprehensive set of multimodal datasets. The extensive experimental results demonstrate Uni-MoE's principal advantage of significantly reducing performance bias in handling mixed multimodal datasets, alongside improved multi-expert collaboration and generalization. Our findings highlight the substantial potential of MoE frameworks in advancing MLLMs and the code is available at https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11157v1">Towards Modular LLMs by Building and Reusing a Library of LoRAs</a></div>
    <div class="paper-meta">
      📅 2024-05-18
    </div>
    <details class="paper-abstract">
      The growing number of parameter-efficient adaptations of a base large language model (LLM) calls for studying whether we can reuse such trained adapters to improve performance for new tasks. We study how to best build a library of adapters given multi-task data and devise techniques for both zero-shot and supervised task generalization through routing in such library. We benchmark existing approaches to build this library and introduce model-based clustering, MBC, a method that groups tasks based on the similarity of their adapter parameters, indirectly optimizing for transfer across the multi-task dataset. To re-use the library, we present a novel zero-shot routing mechanism, Arrow, which enables dynamic selection of the most relevant adapters for new inputs without the need for retraining. We experiment with several LLMs, such as Phi-2 and Mistral, on a wide array of held-out tasks, verifying that MBC-based adapters and Arrow routing lead to superior generalization to new tasks. We make steps towards creating modular, adaptable LLMs that can match or outperform traditional joint training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11106v1">LLM-based Multi-Agent Reinforcement Learning: Current and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 8 pages, 1 figure, 1 table, submitted to IEEE RA-L
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have shown great abilities in various tasks, including question answering, arithmetic problem solving, and poem writing, among others. Although research on LLM-as-an-agent has shown that LLM can be applied to Reinforcement Learning (RL) and achieve decent results, the extension of LLM-based RL to Multi-Agent System (MAS) is not trivial, as many aspects, such as coordination and communication between agents, are not considered in the RL frameworks of a single agent. To inspire more research on LLM-based MARL, in this letter, we survey the existing LLM-based single-agent and multi-agent RL frameworks and provide potential research directions for future research. In particular, we focus on the cooperative tasks of multiple agents with a common goal and communication among them. We also consider human-in/on-the-loop scenarios enabled by the language component in the framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10104v2">GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 Accepted in ACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) and multi-modal models (MMs) have demonstrated their remarkable capabilities in problem-solving. Yet, their proficiency in tackling geometry math problems, which necessitates an integrated understanding of both textual and visual information, has not been thoroughly evaluated. To address this gap, we introduce the GeoEval benchmark, a comprehensive collection that includes a main subset of 2,000 problems, a 750 problems subset focusing on backward reasoning, an augmented subset of 2,000 problems, and a hard subset of 300 problems. This benchmark facilitates a deeper investigation into the performance of LLMs and MMs in solving geometry math problems. Our evaluation of ten LLMs and MMs across these varied subsets reveals that the WizardMath model excels, achieving a 55.67\% accuracy rate on the main subset but only a 6.00\% accuracy on the hard subset. This highlights the critical need for testing models against datasets on which they have not been pre-trained. Additionally, our findings indicate that GPT-series models perform more effectively on problems they have rephrased, suggesting a promising method for enhancing model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09486v1">ENOVA: Autoscaling towards Cost-effective and Stable Serverless LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-05-17
    </div>
    <details class="paper-abstract">
      Since the increasing popularity of large language model (LLM) backend systems, it is common and necessary to deploy stable serverless serving of LLM on multi-GPU clusters with autoscaling. However, there exist challenges because the diversity and co-location of applications in multi-GPU clusters will lead to low service quality and GPU utilization. To address them, we build ENOVA, a deployment, monitoring and autoscaling service towards serverless LLM serving. ENOVA deconstructs the execution process of LLM service comprehensively, based on which ENOVA designs a configuration recommendation module for automatic deployment on any GPU clusters and a performance detection module for autoscaling. On top of them, ENOVA implements a deployment execution engine for multi-GPU cluster scheduling. The experiment results show that ENOVA significantly outperforms other state-of-the-art methods and is suitable for wide deployment in large online systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08949v3">EasyGen: Easing Multimodal Generation with BiDiffuser and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 Accepted by ACL 2024, main conference
    </div>
    <details class="paper-abstract">
      We present EasyGen, an efficient model designed to enhance multimodal understanding and generation by harnessing the capabilities of diffusion models and large language models (LLMs), Unlike existing multimodal models that predominately depend on encoders like CLIP or ImageBind and need ample amounts of training data to bridge modalities,EasyGen leverages BiDiffuser,a bidirectional conditional diffusion model, to foster more efficient modality interactions. Easygen achieves text generation by training a projection layer linking BiDiffuser and an LLM, and facilities image generation by training an adapter to align the LLM's text space with the BiDiffuser's image space, Comprehensive quantitative and qualitative experiments show that EasyGen excels in data-efficient training, high-quality image generation, and extendibility, effectively addressing the challenges in multimodal generation. The source code is available at https://github.com/zxy556677/EasyGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07703v5">OpenLLM-Ro -- Technical Report on Open-source Romanian LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-17
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have achieved almost human-like performance on various tasks. While some LLMs have been trained on multilingual data, most of the training data is in English. Hence, their performance in English greatly exceeds their performance in other languages. This document presents our approach to training and evaluating the first foundational and chat LLM specialized for Romanian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13039v1">Surgical Feature-Space Decomposition of LLMs: Why, When and How?</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 Accepted at ACL 2024
    </div>
    <details class="paper-abstract">
      Low-rank approximations, of the weight and feature space can enhance the performance of deep learning models, whether in terms of improving generalization or reducing the latency of inference. However, there is no clear consensus yet on \emph{how}, \emph{when} and \emph{why} these approximations are helpful for large language models (LLMs). In this work, we empirically study the efficacy of weight and feature space decomposition in transformer-based LLMs. We demonstrate that surgical decomposition not only provides critical insights into the trade-off between compression and language modelling performance, but also sometimes enhances commonsense reasoning performance of LLMs. Our empirical analysis identifies specific network segments that intrinsically exhibit a low-rank structure. Furthermore, we extend our investigation to the implications of low-rank approximations on model bias. Overall, our findings offer a novel perspective on optimizing LLMs, presenting the low-rank approximation not only as a tool for performance enhancements, but also as a means to potentially rectify biases within these models. Our code is available at \href{https://github.com/nyunAI/SFSD-LLM}{GitHub}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13037v1">Enhancing Dialogue State Tracking Models through LLM-backed User-Agents Simulation</a></div>
    <div class="paper-meta">
      📅 2024-05-17
    </div>
    <details class="paper-abstract">
      Dialogue State Tracking (DST) is designed to monitor the evolving dialogue state in the conversations and plays a pivotal role in developing task-oriented dialogue systems. However, obtaining the annotated data for the DST task is usually a costly endeavor. In this paper, we focus on employing LLMs to generate dialogue data to reduce dialogue collection and annotation costs. Specifically, GPT-4 is used to simulate the user and agent interaction, generating thousands of dialogues annotated with DST labels. Then a two-stage fine-tuning on LLaMA 2 is performed on the generated data and the real data for the DST prediction. Experimental results on two public DST benchmarks show that with the generated dialogue data, our model performs better than the baseline trained solely on real data. In addition, our approach is also capable of adapting to the dynamic demands in real-world scenarios, generating dialogues in new domains swiftly. After replacing dialogue segments in any domain with the corresponding generated ones, the model achieves comparable performance to the model trained on real data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.12321v2">Bypassing the Safety Training of Open-Source LLMs with Priming Attacks</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 ICLR Tiny Paper camera ready version
    </div>
    <details class="paper-abstract">
      With the recent surge in popularity of LLMs has come an ever-increasing need for LLM safety training. In this paper, we investigate the fragility of SOTA open-source LLMs under simple, optimization-free attacks we refer to as $\textit{priming attacks}$, which are easy to execute and effectively bypass alignment from safety training. Our proposed attack improves the Attack Success Rate on Harmful Behaviors, as measured by Llama Guard, by up to $3.3\times$ compared to baselines. Source code and data are available at https://github.com/uiuc-focal-lab/llm-priming-attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19021v2">IDGenRec: LLM-RecSys Alignment with Textual ID Learning</a></div>
    <div class="paper-meta">
      📅 2024-05-17
      | 💬 Accepted in SIGIR 2024
    </div>
    <details class="paper-abstract">
      Generative recommendation based on Large Language Models (LLMs) have transformed the traditional ranking-based recommendation style into a text-to-text generation paradigm. However, in contrast to standard NLP tasks that inherently operate on human vocabulary, current research in generative recommendations struggles to effectively encode recommendation items within the text-to-text framework using concise yet meaningful ID representations. To better align LLMs with recommendation needs, we propose IDGen, representing each item as a unique, concise, semantically rich, platform-agnostic textual ID using human language tokens. This is achieved by training a textual ID generator alongside the LLM-based recommender, enabling seamless integration of personalized recommendations into natural language generation. Notably, as user history is expressed in natural language and decoupled from the original dataset, our approach suggests the potential for a foundational generative recommendation model. Experiments show that our framework consistently surpasses existing models in sequential recommendation under standard experimental setting. Then, we explore the possibility of training a foundation recommendation model with the proposed method on data collected from 19 different datasets and tested its recommendation performance on 6 unseen datasets across different platforms under a completely zero-shot setting. The results show that the zero-shot performance of the pre-trained foundation model is comparable to or even better than some traditional recommendation models based on supervised training, showing the potential of the IDGen paradigm serving as the foundation model for generative recommendation. Code and data are open-sourced at https://github.com/agiresearch/IDGenRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10474v1">Rethinking ChatGPT's Success: Usability and Cognitive Behaviors Enabled by Auto-regressive LLMs' Prompting</a></div>
    <div class="paper-meta">
      📅 2024-05-17
    </div>
    <details class="paper-abstract">
      Over the last decade, a wide range of training and deployment strategies for Large Language Models (LLMs) have emerged. Among these, the prompting paradigms of Auto-regressive LLMs (AR-LLMs) have catalyzed a significant surge in Artificial Intelligence (AI). This paper aims to emphasize the significance of utilizing free-form modalities (forms of input and output) and verbal free-form contexts as user-directed channels (methods for transforming modalities) for downstream deployment. Specifically, we analyze the structure of modalities within both two types of LLMs and six task-specific channels during deployment. From the perspective of users, our analysis introduces and applies the analytical metrics of task customizability, transparency, and complexity to gauge their usability, highlighting the superior nature of AR-LLMs' prompting paradigms. Moreover, we examine the stimulation of diverse cognitive behaviors in LLMs through the adoption of free-form text and verbal contexts, mirroring human linguistic expressions of such behaviors. We then detail four common cognitive behaviors to underscore how AR-LLMs' prompting successfully imitate human-like behaviors using this free-form modality and channel. Lastly, the potential for improving LLM deployment, both as autonomous agents and within multi-agent systems, is identified via cognitive behavior concepts and principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13036v1">Can formal argumentative reasoning enhance LLMs performances?</a></div>
    <div class="paper-meta">
      📅 2024-05-16
    </div>
    <details class="paper-abstract">
      Recent years witnessed significant performance advancements in deep-learning-driven natural language models, with a strong focus on the development and release of Large Language Models (LLMs). These improvements resulted in better quality AI-generated output but rely on resource-expensive training and upgrading of models. Although different studies have proposed a range of techniques to enhance LLMs without retraining, none have considered computational argumentation as an option. This is a missed opportunity since computational argumentation is an intuitive mechanism that formally captures agents' interactions and the information conflict that may arise during such interplays, and so it seems well-suited for boosting the reasoning and conversational abilities of LLMs in a seamless manner. In this paper, we present a pipeline (MQArgEng) and preliminary study to evaluate the effect of introducing computational argumentation semantics on the performance of LLMs. Our experiment's goal was to provide a proof-of-concept and a feasibility analysis in order to foster (or deter) future research towards a fully-fledged argumentation engine plugin for LLMs. Exploratory results using the MT-Bench indicate that MQArgEng provides a moderate performance gain in most of the examined topical categories and, as such, show promise and warrant further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10255v1">When LLMs step into the 3D World: A Survey and Meta-Analysis of 3D Tasks via Multi-modal Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-05-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve, their integration with 3D spatial data (3D-LLMs) has seen rapid progress, offering unprecedented capabilities for understanding and interacting with physical spaces. This survey provides a comprehensive overview of the methodologies enabling LLMs to process, understand, and generate 3D data. Highlighting the unique advantages of LLMs, such as in-context learning, step-by-step reasoning, open-vocabulary capabilities, and extensive world knowledge, we underscore their potential to significantly advance spatial comprehension and interaction within embodied Artificial Intelligence (AI) systems. Our investigation spans various 3D data representations, from point clouds to Neural Radiance Fields (NeRFs). It examines their integration with LLMs for tasks such as 3D scene understanding, captioning, question-answering, and dialogue, as well as LLM-based agents for spatial reasoning, planning, and navigation. The paper also includes a brief review of other methods that integrate 3D and language. The meta-analysis presented in this paper reveals significant progress yet underscores the necessity for novel approaches to harness the full potential of 3D-LLMs. Hence, with this paper, we aim to chart a course for future research that explores and expands the capabilities of 3D-LLMs in understanding and interacting with the complex 3D world. To support this survey, we have established a project page where papers related to our topic are organized and listed: https://github.com/ActiveVisionLab/Awesome-LLM-3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08997v2">LLM-Assisted Rule Based Machine Translation for Low/No-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2024-05-16
    </div>
    <details class="paper-abstract">
      We propose a new paradigm for machine translation that is particularly useful for no-resource languages (those without any publicly available bilingual or monolingual corpora): LLM-RBMT (LLM-Assisted Rule Based Machine Translation). Using the LLM-RBMT paradigm, we design the first language education/revitalization-oriented machine translator for Owens Valley Paiute (OVP), a critically endangered Indigenous American language for which there is virtually no publicly available data. We present a detailed evaluation of the translator's components: a rule-based sentence builder, an OVP to English translator, and an English to OVP translator. We also discuss the potential of the paradigm, its limitations, and the many avenues for future research that it opens up.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10121v1">Distilling Implicit Multimodal Knowledge into LLMs for Zero-Resource Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-16
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Integrating multimodal knowledge into large language models (LLMs) represents a significant advancement in dialogue generation capabilities. However, the effective incorporation of such knowledge in zero-resource scenarios remains a substantial challenge due to the scarcity of diverse, high-quality dialogue datasets. To address this, we propose the Visual Implicit Knowledge Distillation Framework (VIKDF), an innovative approach aimed at enhancing LLMs for enriched dialogue generation in zero-resource contexts by leveraging implicit multimodal knowledge. VIKDF comprises two main stages: knowledge distillation, using an Implicit Query Transformer to extract and encode visual implicit knowledge from image-text pairs into knowledge vectors; and knowledge integration, employing a novel Bidirectional Variational Information Fusion technique to seamlessly integrate these distilled vectors into LLMs. This enables the LLMs to generate dialogues that are not only coherent and engaging but also exhibit a deep understanding of the context through implicit multimodal cues, effectively overcoming the limitations of zero-resource scenarios. Our extensive experimentation across two dialogue datasets shows that VIKDF outperforms existing state-of-the-art models in generating high-quality dialogues. The code will be publicly available following acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00029v1">Distributed Inference Performance Optimization for LLMs on CPUs</a></div>
    <div class="paper-meta">
      📅 2024-05-16
      | 💬 4 pages, 3 figures, Practical ML for Low Resource Settings Workshop @ ICLR 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) hold tremendous potential for addressing numerous real-world challenges, yet they typically demand significant computational resources and memory. Deploying LLMs onto a resource-limited hardware device with restricted memory capacity presents considerable challenges. Distributed computing emerges as a prevalent strategy to mitigate single-node memory constraints and expedite LLM inference performance. To reduce the hardware limitation burden, we proposed an efficient distributed inference optimization solution for LLMs on CPUs. We conduct experiments with the proposed solution on 5th Gen Intel Xeon Scalable Processors, and the result shows the time per output token for the LLM with 72B parameter is 140 ms/token, much faster than the average human reading speed about 200ms per token.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09783v1">LLM and Simulation as Bilevel Optimizers: A New Paradigm to Advance Physical Scientific Discovery</a></div>
    <div class="paper-meta">
      📅 2024-05-16
      | 💬 ICML 2024
    </div>
    <details class="paper-abstract">
      Large Language Models have recently gained significant attention in scientific discovery for their extensive knowledge and advanced reasoning capabilities. However, they encounter challenges in effectively simulating observational feedback and grounding it with language to propel advancements in physical scientific discovery. Conversely, human scientists undertake scientific discovery by formulating hypotheses, conducting experiments, and revising theories through observational analysis. Inspired by this, we propose to enhance the knowledge-driven, abstract reasoning abilities of LLMs with the computational strength of simulations. We introduce Scientific Generative Agent (SGA), a bilevel optimization framework: LLMs act as knowledgeable and versatile thinkers, proposing scientific hypotheses and reason about discrete components, such as physics equations or molecule structures; meanwhile, simulations function as experimental platforms, providing observational feedback and optimizing via differentiability for continuous parts, such as physical parameters. We conduct extensive experiments to demonstrate our framework's efficacy in constitutive law discovery and molecular design, unveiling novel solutions that differ from conventional human expectations yet remain coherent upon analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04291v2">BiLLM: Pushing the Limit of Post-Training Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-15
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Pretrained large language models (LLMs) exhibit exceptional general language processing capabilities but come with significant demands on memory and computational resources. As a powerful compression technology, binarization can extremely reduce model weights to a mere 1 bit, lowering the expensive computation and memory requirements. However, existing quantization techniques fall short of maintaining LLM performance under ultra-low bit-widths. In response to this challenge, we present BiLLM, a groundbreaking 1-bit post-training quantization scheme tailored for pretrained LLMs. Based on the weight distribution of LLMs, BiLLM first identifies and structurally selects salient weights, and minimizes the compression loss through an effective binary residual approximation strategy. Moreover, considering the bell-shaped distribution of the non-salient weights, we propose an optimal splitting search to group and binarize them accurately. BiLLM achieving for the first time high-accuracy inference (e.g. 8.41 perplexity on LLaMA2-70B) with only 1.08-bit weights across various LLMs families and evaluation metrics, outperforms SOTA quantization methods of LLM by significant margins. Moreover, BiLLM enables the binarization process of the LLM with 7 billion weights within 0.5 hours on a single GPU, demonstrating satisfactory time efficiency. Our code is available at https://github.com/Aaronhuang-778/BiLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13020v1">Using Combinatorial Optimization to Design a High quality LLM Solution</a></div>
    <div class="paper-meta">
      📅 2024-05-15
    </div>
    <details class="paper-abstract">
      We introduce a novel LLM based solution design approach that utilizes combinatorial optimization and sampling. Specifically, a set of factors that influence the quality of the solution are identified. They typically include factors that represent prompt types, LLM inputs alternatives, and parameters governing the generation and design alternatives. Identifying the factors that govern the LLM solution quality enables the infusion of subject matter expert knowledge. Next, a set of interactions between the factors are defined and combinatorial optimization is used to create a small subset $P$ that ensures all desired interactions occur in $P$. Each element $p \in P$ is then developed into an appropriate benchmark. Applying the alternative solutions on each combination, $p \in P$ and evaluating the results facilitate the design of a high quality LLM solution pipeline. The approach is especially applicable when the design and evaluation of each benchmark in $P$ is time-consuming and involves manual steps and human evaluation. Given its efficiency the approach can also be used as a baseline to compare and validate an autoML approach that searches over the factors governing the solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09113v1">Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization</a></div>
    <div class="paper-meta">
      📅 2024-05-15
    </div>
    <details class="paper-abstract">
      Recent research indicates that large language models (LLMs) are susceptible to jailbreaking attacks that can generate harmful content. This paper introduces a novel token-level attack method, Adaptive Dense-to-Sparse Constrained Optimization (ADC), which effectively jailbreaks several open-source LLMs. Our approach relaxes the discrete jailbreak optimization into a continuous optimization and progressively increases the sparsity of the optimizing vectors. Consequently, our method effectively bridges the gap between discrete and continuous space optimization. Experimental results demonstrate that our method is more effective and efficient than existing token-level methods. On Harmbench, our method achieves state of the art attack success rate on seven out of eight LLMs. Code will be made available. Trigger Warning: This paper contains model behavior that can be offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.09090v1">Towards Next-Generation Steganalysis: LLMs Unleash the Power of Detecting Steganography</a></div>
    <div class="paper-meta">
      📅 2024-05-15
    </div>
    <details class="paper-abstract">
      Linguistic steganography provides convenient implementation to hide messages, particularly with the emergence of AI generation technology. The potential abuse of this technology raises security concerns within societies, calling for powerful linguistic steganalysis to detect carrier containing steganographic messages. Existing methods are limited to finding distribution differences between steganographic texts and normal texts from the aspect of symbolic statistics. However, the distribution differences of both kinds of texts are hard to build precisely, which heavily hurts the detection ability of the existing methods in realistic scenarios. To seek a feasible way to construct practical steganalysis in real world, this paper propose to employ human-like text processing abilities of large language models (LLMs) to realize the difference from the aspect of human perception, addition to traditional statistic aspect. Specifically, we systematically investigate the performance of LLMs in this task by modeling it as a generative paradigm, instead of traditional classification paradigm. Extensive experiment results reveal that generative LLMs exhibit significant advantages in linguistic steganalysis and demonstrate performance trends distinct from traditional approaches. Results also reveal that LLMs outperform existing baselines by a wide margin, and the domain-agnostic ability of LLMs makes it possible to train a generic steganalysis model (Both codes and trained models are openly available in https://github.com/ba0z1/Linguistic-Steganalysis-with-LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06725v3">On the Shape of Brainscores for Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-05-15
      | 💬 Published as a workshop paper at ICLR AGI Workshop 2024
    </div>
    <details class="paper-abstract">
      With the rise of Large Language Models (LLMs), the novel metric "Brainscore" emerged as a means to evaluate the functional similarity between LLMs and human brain/neural systems. Our efforts were dedicated to mining the meaning of the novel score by constructing topological features derived from both human fMRI data involving 190 subjects, and 39 LLMs plus their untrained counterparts. Subsequently, we trained 36 Linear Regression Models and conducted thorough statistical analyses to discern reliable and valid features from our constructed ones. Our findings reveal distinctive feature combinations conducive to interpreting existing brainscores across various brain regions of interest (ROIs) and hemispheres, thereby significantly contributing to advancing interpretable machine learning (iML) studies. The study is enriched by our further discussions and analyses concerning existing brainscores. To our knowledge, this study represents the first attempt to comprehend the novel metric brainscore within this interdisciplinary domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.01974v2">Towards Truly Zero-shot Compositional Visual Reasoning with LLMs as Programmers</a></div>
    <div class="paper-meta">
      📅 2024-05-14
    </div>
    <details class="paper-abstract">
      Visual reasoning is dominated by end-to-end neural networks scaled to billions of model parameters and training examples. However, even the largest models struggle with compositional reasoning, generalization, fine-grained spatial and temporal reasoning, and counting. Visual reasoning with large language models (LLMs) as controllers can, in principle, address these limitations by decomposing the task and solving subtasks by orchestrating a set of (visual) tools. Recently, these models achieved great performance on tasks such as compositional visual question answering, visual grounding, and video temporal reasoning. Nevertheless, in their current form, these models heavily rely on human engineering of in-context examples in the prompt, which are often dataset- and task-specific and require significant labor by highly skilled programmers. In this work, we present a framework that mitigates these issues by introducing spatially and temporally abstract routines and by leveraging a small number of labeled examples to automatically generate in-context examples, thereby avoiding human-created in-context examples. On a number of visual reasoning tasks, we show that our framework leads to consistent gains in performance, makes LLMs as controllers setup more robust, and removes the need for human engineering of in-context examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08792v1">Towards Enhanced RAC Accessibility: Leveraging Datasets and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-14
    </div>
    <details class="paper-abstract">
      This paper explores the potential of large language models (LLMs) to make the Aeronautical Regulations of Colombia (RAC) more accessible. Given the complexity and extensive technicality of the RAC, this study introduces a novel approach to simplifying these regulations for broader understanding. By developing the first-ever RAC database, which contains 24,478 expertly labeled question-and-answer pairs, and fine-tuning LLMs specifically for RAC applications, the paper outlines the methodology for dataset assembly, expert-led annotation, and model training. Utilizing the Gemma1.1 2b model along with advanced techniques like Unsloth for efficient VRAM usage and flash attention mechanisms, the research aims to expedite training processes. This initiative establishes a foundation to enhance the comprehensibility and accessibility of RAC, potentially benefiting novices and reducing dependence on expert consultations for navigating the aviation industry's regulatory landscape. You can visit the dataset (https://huggingface.co/somosnlp/gemma-1.1-2b-it_ColombiaRAC_FullyCurated_format_chatML_V1) and the model (https://huggingface.co/datasets/somosnlp/ColombiaRAC_FullyCurated) here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08502v1">Archimedes-AUEB at SemEval-2024 Task 5: LLM explains Civil Procedure</a></div>
    <div class="paper-meta">
      📅 2024-05-14
      | 💬 To be published in SemEval-2024
    </div>
    <details class="paper-abstract">
      The SemEval task on Argument Reasoning in Civil Procedure is challenging in that it requires understanding legal concepts and inferring complex arguments. Currently, most Large Language Models (LLM) excelling in the legal realm are principally purposed for classification tasks, hence their reasoning rationale is subject to contention. The approach we advocate involves using a powerful teacher-LLM (ChatGPT) to extend the training dataset with explanations and generate synthetic data. The resulting data are then leveraged to fine-tune a small student-LLM. Contrary to previous work, our explanations are not directly derived from the teacher's internal knowledge. Instead they are grounded in authentic human analyses, therefore delivering a superior reasoning signal. Additionally, a new `mutation' method generates artificial data instances inspired from existing ones. We are publicly releasing the explanations as an extension to the original dataset, along with the synthetic dataset and the prompts that were used to generate both. Our system ranked 15th in the SemEval competition. It outperforms its own teacher and can produce explanations aligned with the original human analyses, as verified by legal experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08839v1">PromptMind Team at EHRSQL-2024: Improving Reliability of SQL Generation using Ensemble LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-14
      | 💬 Accepted as a poster for Clinical NLP workshop at NAACL 2024
    </div>
    <details class="paper-abstract">
      This paper presents our approach to the EHRSQL-2024 shared task, which aims to develop a reliable Text-to-SQL system for electronic health records. We propose two approaches that leverage large language models (LLMs) for prompting and fine-tuning to generate EHRSQL queries. In both techniques, we concentrate on bridging the gap between the real-world knowledge on which LLMs are trained and the domain specific knowledge required for the task. The paper provides the results of each approach individually, demonstrating that they achieve high execution accuracy. Additionally, we show that an ensemble approach further enhances generation reliability by reducing errors. This approach secured us 2nd place in the shared task competition. The methodologies outlined in this paper are designed to be transferable to domain-specific Text-to-SQL problems that emphasize both accuracy and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08373v1">PromptMind Team at MEDIQA-CORR 2024: Improving Clinical Text Correction with Error Categorization and LLM Ensembles</a></div>
    <div class="paper-meta">
      📅 2024-05-14
      | 💬 Paper accepted for oral presentation at Clinical NLP workshop, NAACL 2024
    </div>
    <details class="paper-abstract">
      This paper describes our approach to the MEDIQA-CORR shared task, which involves error detection and correction in clinical notes curated by medical professionals. This task involves handling three subtasks: detecting the presence of errors, identifying the specific sentence containing the error, and correcting it. Through our work, we aim to assess the capabilities of Large Language Models (LLMs) trained on a vast corpora of internet data that contain both factual and unreliable information. We propose to comprehensively address all subtasks together, and suggest employing a unique prompt-based in-context learning strategy. We will evaluate its efficacy in this specialized task demanding a combination of general reasoning and medical knowledge. In medical systems where prediction errors can have grave consequences, we propose leveraging self-consistency and ensemble methods to enhance error correction and error detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15151v2">Where Visual Speech Meets Language: VSP-LLM Framework for Efficient and Context-Aware Visual Speech Processing</a></div>
    <div class="paper-meta">
      📅 2024-05-14
      | 💬 An Erratum was added on the last page of this paper
    </div>
    <details class="paper-abstract">
      In visual speech processing, context modeling capability is one of the most important requirements due to the ambiguous nature of lip movements. For example, homophenes, words that share identical lip movements but produce different sounds, can be distinguished by considering the context. In this paper, we propose a novel framework, namely Visual Speech Processing incorporated with LLMs (VSP-LLM), to maximize the context modeling ability by bringing the overwhelming power of LLMs. Specifically, VSP-LLM is designed to perform multi-tasks of visual speech recognition and translation, where the given instructions control the type of task. The input video is mapped to the input latent space of an LLM by employing a self-supervised visual speech model. Focused on the fact that there is redundant information in input frames, we propose a novel deduplication method that reduces the embedded visual features by employing visual speech units. Through the proposed deduplication and Low Rank Adaptation (LoRA), VSP-LLM can be trained in a computationally efficient manner. In the translation dataset, the MuAViC benchmark, we demonstrate that VSP-LLM trained on just 30 hours of labeled data can more effectively translate lip movements compared to the recent model trained with 433 hours of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08154v1">LLM Theory of Mind and Alignment: Opportunities and Risks</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming human-computer interaction and conceptions of artificial intelligence (AI) with their impressive capacities for conversing and reasoning in natural language. There is growing interest in whether LLMs have theory of mind (ToM); the ability to reason about the mental and emotional states of others that is core to human social intelligence. As LLMs are integrated into the fabric of our personal, professional and social lives and given greater agency to make decisions with real-world consequences, there is a critical need to understand how they can be aligned with human values. ToM seems to be a promising direction of inquiry in this regard. Following the literature on the role and impacts of human ToM, this paper identifies key areas in which LLM ToM will show up in human:LLM interactions at individual and group levels, and what opportunities and risks for alignment are raised in each. On the individual level, the paper considers how LLM ToM might manifest in goal specification, conversational adaptation, empathy and anthropomorphism. On the group level, it considers how LLM ToM might facilitate collective alignment, cooperation or competition, and moral judgement-making. The paper lays out a broad spectrum of potential implications and suggests the most pressing areas for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00858v4">Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-13
      | 💬 8 pages, 3 figures, Published at the ICLR 2024 Workshop on Understanding of Foundation Models (ME-FoMo)
    </div>
    <details class="paper-abstract">
      Text generation with Large Language Models (LLMs) is known to be memory bound due to the combination of their auto-regressive nature, huge parameter counts, and limited memory bandwidths, often resulting in low token rates. Speculative decoding has been proposed as a solution for LLM inference acceleration. However, since draft models are often unavailable in the modern open-source LLM families, e.g., for Llama 2 7B, training a high-quality draft model is required to enable inference acceleration via speculative decoding. In this paper, we propose a simple draft model training framework for direct alignment to chat-capable target models. With the proposed framework, we train Llama 2 Chat Drafter 115M, a draft model for Llama 2 Chat 7B or larger, with only 1.64\% of the original size. Our training framework only consists of pretraining, distillation dataset generation, and finetuning with knowledge distillation, with no additional alignment procedure. For the finetuning step, we use instruction-response pairs generated by target model for distillation in plausible data distribution, and propose a new Total Variation Distance++ (TVD++) loss that incorporates variance reduction techniques inspired from the policy gradient method in reinforcement learning. Our empirical results show that Llama 2 Chat Drafter 115M with speculative decoding achieves up to 2.3 block efficiency and 2.4$\times$ speed-up relative to autoregressive decoding on various tasks with no further task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05329v2">KV-Runahead: Scalable Causal LLM Inference by Parallel Key-Value Cache Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-13
      | 💬 preprint for ICML 2024
    </div>
    <details class="paper-abstract">
      Large Language Model or LLM inference has two phases, the prompt (or prefill) phase to output the first token and the extension (or decoding) phase to the generate subsequent tokens. In this work, we propose an efficient parallelization scheme, KV-Runahead to accelerate the prompt phase. The key observation is that the extension phase generates tokens faster than the prompt phase because of key-value cache (KV-cache). Hence, KV-Runahead parallelizes the prompt phase by orchestrating multiple processes to populate the KV-cache and minimizes the time-to-first-token (TTFT). Dual-purposing the KV-cache scheme has two main benefits. First, since KV-cache is designed to leverage the causal attention map, we minimize computation and computation automatically. Second, since it already exists for the extension phase, KV-Runahead is easy to implement. We further propose context-level load-balancing to handle uneven KV-cache generation (due to the causal attention) and to optimize TTFT. Compared with an existing parallelization scheme such as tensor or sequential parallelization where keys and values are locally generated and exchanged via all-gather collectives, our experimental results demonstrate that KV-Runahead can offer over 1.4x and 1.6x speedups for Llama 7B and Falcon 7B respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07840v1">Open-vocabulary Auditory Neural Decoding Using fMRI-prompted LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      Decoding language information from brain signals represents a vital research area within brain-computer interfaces, particularly in the context of deciphering the semantic information from the fMRI signal. However, many existing efforts concentrate on decoding small vocabulary sets, leaving space for the exploration of open vocabulary continuous text decoding. In this paper, we introduce a novel method, the \textbf{Brain Prompt GPT (BP-GPT)}. By using the brain representation that is extracted from the fMRI as a prompt, our method can utilize GPT-2 to decode fMRI signals into stimulus text. Further, we introduce a text-to-text baseline and align the fMRI prompt to the text prompt. By introducing the text-to-text baseline, our BP-GPT can extract a more robust brain prompt and promote the decoding of pre-trained LLM. We evaluate our BP-GPT on the open-source auditory semantic decoding dataset and achieve a significant improvement up to $4.61\%$ on METEOR and $2.43\%$ on BERTScore across all the subjects compared to the state-of-the-art method. The experimental results demonstrate that using brain representation as a prompt to further drive LLM for auditory neural decoding is feasible and effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07828v1">Can LLMs Help Predict Elections? (Counter)Evidence from the World's Largest Democracy</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      The study of how social media affects the formation of public opinion and its influence on political results has been a popular field of inquiry. However, current approaches frequently offer a limited comprehension of the complex political phenomena, yielding inconsistent outcomes. In this work, we introduce a new method: harnessing the capabilities of Large Language Models (LLMs) to examine social media data and forecast election outcomes. Our research diverges from traditional methodologies in two crucial respects. First, we utilize the sophisticated capabilities of foundational LLMs, which can comprehend the complex linguistic subtleties and contextual details present in social media data. Second, we focus on data from X (Twitter) in India to predict state assembly election outcomes. Our method entails sentiment analysis of election-related tweets through LLMs to forecast the actual election results, and we demonstrate the superiority of our LLM-based method against more traditional exit and opinion polls. Overall, our research offers valuable insights into the unique dynamics of Indian politics and the remarkable impact of social media in molding public attitudes within this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05099v2">Hydragen: High-Throughput LLM Inference with Shared Prefixes</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) are now deployed to hundreds of millions of users. LLM inference is commonly performed on batches of sequences that share a prefix, such as few-shot examples or a chatbot system prompt. Decoding in this large-batch setting can be bottlenecked by the attention operation, which reads large key-value (KV) caches from memory and computes inefficient matrix-vector products for every sequence in the batch. In this work, we introduce Hydragen, a hardware-aware exact implementation of attention with shared prefixes. Hydragen computes attention over the shared prefix and unique suffixes separately. This decomposition enables efficient prefix attention by batching queries together across sequences, reducing redundant memory reads and enabling the use of hardware-friendly matrix multiplications. Our method can improve end-to-end CodeLlama-13b throughput by up to 32x against competitive baselines, with speedup growing with the batch size and shared prefix length. Hydragen also enables the use of very long shared contexts: with a large batch size, increasing the prefix length from 1K to 16K tokens decreases Hydragen throughput by less than 15%, while the throughput of baselines drops by over 90%. Hydragen generalizes beyond simple prefix-suffix decomposition and can be applied to tree-based prompt sharing patterns, allowing us to further reduce inference time on competitive programming problems by 55%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07496v1">Oedipus: LLM-enchanced Reasoning CAPTCHA Solver</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      CAPTCHAs have become a ubiquitous tool in safeguarding applications from automated bots. Over time, the arms race between CAPTCHA development and evasion techniques has led to increasingly sophisticated and diverse designs. The latest iteration, reasoning CAPTCHAs, exploits tasks that are intuitively simple for humans but challenging for conventional AI technologies, thereby enhancing security measures. Driven by the evolving AI capabilities, particularly the advancements in Large Language Models (LLMs), we investigate the potential of multimodal LLMs to solve modern reasoning CAPTCHAs. Our empirical analysis reveals that, despite their advanced reasoning capabilities, LLMs struggle to solve these CAPTCHAs effectively. In response, we introduce Oedipus, an innovative end-to-end framework for automated reasoning CAPTCHA solving. Central to this framework is a novel strategy that dissects the complex and human-easy-AI-hard tasks into a sequence of simpler and AI-easy steps. This is achieved through the development of a Domain Specific Language (DSL) for CAPTCHAs that guides LLMs in generating actionable sub-steps for each CAPTCHA challenge. The DSL is customized to ensure that each unit operation is a highly solvable subtask revealed in our previous empirical study. These sub-steps are then tackled sequentially using the Chain-of-Thought (CoT) methodology. Our evaluation shows that Oedipus effectively resolves the studied CAPTCHAs, achieving an average success rate of 63.5\%. Remarkably, it also shows adaptability to the most recent CAPTCHA designs introduced in late 2023, which are not included in our initial study. This prompts a discussion on future strategies for designing reasoning CAPTCHAs that can effectively counter advanced AI solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08035v1">A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-05-13
    </div>
    <details class="paper-abstract">
      Conversational Recommender System (CRS) leverages real-time feedback from users to dynamically model their preferences, thereby enhancing the system's ability to provide personalized recommendations and improving the overall user experience. CRS has demonstrated significant promise, prompting researchers to concentrate their efforts on developing user simulators that are both more realistic and trustworthy. The emergence of Large Language Models (LLMs) has marked the onset of a new epoch in computational capabilities, exhibiting human-level intelligence in various tasks. Research efforts have been made to utilize LLMs for building user simulators to evaluate the performance of CRS. Although these efforts showcase innovation, they are accompanied by certain limitations. In this work, we introduce a Controllable, Scalable, and Human-Involved (CSHI) simulator framework that manages the behavior of user simulators across various stages via a plugin manager. CSHI customizes the simulation of user behavior and interactions to provide a more lifelike and convincing user interaction experience. Through experiments and case studies in two conversational recommendation scenarios, we show that our framework can adapt to a variety of conversational recommendation settings and effectively simulate users' personalized preferences. Consequently, our simulator is able to generate feedback that closely mirrors that of real users. This facilitates a reliable assessment of existing CRS studies and promotes the creation of high-quality conversational recommendation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07320v1">L(u)PIN: LLM-based Political Ideology Nowcasting</a></div>
    <div class="paper-meta">
      📅 2024-05-12
    </div>
    <details class="paper-abstract">
      The quantitative analysis of political ideological positions is a difficult task. In the past, various literature focused on parliamentary voting data of politicians, party manifestos and parliamentary speech to estimate political disagreement and polarization in various political systems. However previous methods of quantitative political analysis suffered from a common challenge which was the amount of data available for analysis. Also previous methods frequently focused on a more general analysis of politics such as overall polarization of the parliament or party-wide political ideological positions. In this paper, we present a method to analyze ideological positions of individual parliamentary representatives by leveraging the latent knowledge of LLMs. The method allows us to evaluate the stance of politicians on an axis of our choice allowing us to flexibly measure the stance of politicians in regards to a topic/controversy of our choice. We achieve this by using a fine-tuned BERT classifier to extract the opinion-based sentences from the speeches of representatives and projecting the average BERT embeddings for each representative on a pair of reference seeds. These reference seeds are either manually chosen representatives known to have opposing views on a particular topic or they are generated sentences which where created using the GPT-4 model of OpenAI. We created the sentences by prompting the GPT-4 model to generate a speech that would come from a politician defending a particular position.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02178v2">Assessing and Verifying Task Utility in LLM-Powered Applications</a></div>
    <div class="paper-meta">
      📅 2024-05-12
      | 💬 arXiv admin note: text overlap with arXiv:2402.09015
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has led to a surge in applications that facilitate collaboration among multiple agents, assisting humans in their daily tasks. However, a significant gap remains in assessing to what extent LLM-powered applications genuinely enhance user experience and task execution efficiency. This highlights the need to verify utility of LLM-powered applications, particularly by ensuring alignment between the application's functionality and end-user needs. We introduce AgentEval, a novel framework designed to simplify the utility verification process by automatically proposing a set of criteria tailored to the unique purpose of any given application. This allows for a comprehensive assessment, quantifying the utility of an application against the suggested criteria. We present a comprehensive analysis of the effectiveness and robustness of AgentEval for two open source datasets including Math Problem solving and ALFWorld House-hold related tasks. For reproducibility purposes, we make the data, code and all the logs publicly available at https://bit.ly/3w3yKcS .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07248v1">Limited Ability of LLMs to Simulate Human Psychological Behaviours: a Psychometric Analysis</a></div>
    <div class="paper-meta">
      📅 2024-05-12
    </div>
    <details class="paper-abstract">
      The humanlike responses of large language models (LLMs) have prompted social scientists to investigate whether LLMs can be used to simulate human participants in experiments, opinion polls and surveys. Of central interest in this line of research has been mapping out the psychological profiles of LLMs by prompting them to respond to standardized questionnaires. The conflicting findings of this research are unsurprising given that mapping out underlying, or latent, traits from LLMs' text responses to questionnaires is no easy task. To address this, we use psychometrics, the science of psychological measurement. In this study, we prompt OpenAI's flagship models, GPT-3.5 and GPT-4, to assume different personas and respond to a range of standardized measures of personality constructs. We used two kinds of persona descriptions: either generic (four or five random person descriptions) or specific (mostly demographics of actual humans from a large-scale human dataset). We found that the responses from GPT-4, but not GPT-3.5, using generic persona descriptions show promising, albeit not perfect, psychometric properties, similar to human norms, but the data from both LLMs when using specific demographic profiles, show poor psychometrics properties. We conclude that, currently, when LLMs are asked to simulate silicon personas, their responses are poor signals of potentially underlying latent traits. Thus, our work casts doubt on LLMs' ability to simulate individual-level human behaviour across multiple-choice question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07212v1">Enhancing Decision-Making in Optimization through LLM-Assisted Inference: A Neural Networks Perspective</a></div>
    <div class="paper-meta">
      📅 2024-05-12
      | 💬 Accepted IJCNN
    </div>
    <details class="paper-abstract">
      This paper explores the seamless integration of Generative AI (GenAI) and Evolutionary Algorithms (EAs) within the domain of large-scale multi-objective optimization. Focusing on the transformative role of Large Language Models (LLMs), our study investigates the potential of LLM-Assisted Inference to automate and enhance decision-making processes. Specifically, we highlight its effectiveness in illuminating key decision variables in evolutionarily optimized solutions while articulating contextual trade-offs. Tailored to address the challenges inherent in inferring complex multi-objective optimization solutions at scale, our approach emphasizes the adaptive nature of LLMs, allowing them to provide nuanced explanations and align their language with diverse stakeholder expertise levels and domain preferences. Empirical studies underscore the practical applicability and impact of LLM-Assisted Inference in real-world decision-making scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07111v1">Designing and Evaluating Dialogue LLMs for Co-Creative Improvised Theatre</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 13 pages, 7 figures, accepted for publication at the International Conference on Computational Creativity 2024
    </div>
    <details class="paper-abstract">
      Social robotics researchers are increasingly interested in multi-party trained conversational agents. With a growing demand for real-world evaluations, our study presents Large Language Models (LLMs) deployed in a month-long live show at the Edinburgh Festival Fringe. This case study investigates human improvisers co-creating with conversational agents in a professional theatre setting. We explore the technical capabilities and constraints of on-the-spot multi-party dialogue, providing comprehensive insights from both audience and performer experiences with AI on stage. Our human-in-the-loop methodology underlines the challenges of these LLMs in generating context-relevant responses, stressing the user interface's crucial role. Audience feedback indicates an evolving interest for AI-driven live entertainment, direct human-AI interaction, and a diverse range of expectations about AI's conversational competence and utility as a creativity support tool. Human performers express immense enthusiasm, varied satisfaction, and the evolving public opinion highlights mixed emotions about AI's role in arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13000v1">RAGE Against the Machine: Retrieval-Augmented LLM Explanations</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 Accepted by ICDE 2024 (Demonstration Track)
    </div>
    <details class="paper-abstract">
      This paper demonstrates RAGE, an interactive tool for explaining Large Language Models (LLMs) augmented with retrieval capabilities; i.e., able to query external sources and pull relevant information into their input context. Our explanations are counterfactual in the sense that they identify parts of the input context that, when removed, change the answer to the question posed to the LLM. RAGE includes pruning methods to navigate the vast space of possible explanations, allowing users to view the provenance of the produced answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04645v2">Enhancing LLM-Based Feedback: Insights from Intelligent Tutoring Systems and the Learning Sciences</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 Accepted to 25th International Conference on Artificial Intelligence in Education (AIED 2024) BlueSky special track
    </div>
    <details class="paper-abstract">
      The field of Artificial Intelligence in Education (AIED) focuses on the intersection of technology, education, and psychology, placing a strong emphasis on supporting learners' needs with compassion and understanding. The growing prominence of Large Language Models (LLMs) has led to the development of scalable solutions within educational settings, including generating different types of feedback in Intelligent Tutoring Systems. However, the approach to utilizing these models often involves directly formulating prompts to solicit specific information, lacking a solid theoretical foundation for prompt construction and empirical assessments of their impact on learning. This work advocates careful and caring AIED research by going through previous research on feedback generation in ITS, with emphasis on the theoretical frameworks they utilized and the efficacy of the corresponding design in empirical evaluations, and then suggesting opportunities to apply these evidence-based principles to the design, experiment, and evaluation phases of LLM-based feedback generation. The main contributions of this paper include: an avocation of applying more cautious, theoretically grounded methods in feedback generation in the era of generative AI; and practical suggestions on theory and evidence-based feedback design for LLM-powered ITS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08017v1">Translating Expert Intuition into Quantifiable Features: Encode Investigator Domain Knowledge via LLM for Enhanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2024-05-11
    </div>
    <details class="paper-abstract">
      In the realm of predictive analytics, the nuanced domain knowledge of investigators often remains underutilized, confined largely to subjective interpretations and ad hoc decision-making. This paper explores the potential of Large Language Models (LLMs) to bridge this gap by systematically converting investigator-derived insights into quantifiable, actionable features that enhance model performance. We present a framework that leverages LLMs' natural language understanding capabilities to encode these red flags into a structured feature set that can be readily integrated into existing predictive models. Through a series of case studies, we demonstrate how this approach not only preserves the critical human expertise within the investigative process but also scales the impact of this knowledge across various prediction tasks. The results indicate significant improvements in risk assessment and decision-making accuracy, highlighting the value of blending human experiential knowledge with advanced machine learning techniques. This study paves the way for more sophisticated, knowledge-driven analytics in fields where expert insight is paramount.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10371v1">The Silent Curriculum: How Does LLM Monoculture Shape Educational Content and Its Accessibility?</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 5 pages and 4 figures. Accepted at The Workshop on Global AI Cultures at the International Conference on Learning Representations, 2024 (ICLR'24)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) ascend in popularity, offering information with unprecedented convenience compared to traditional search engines, we delve into the intriguing possibility that a new, singular perspective is being propagated. We call this the "Silent Curriculum," where our focus shifts towards a particularly impressionable demographic: children, who are drawn to the ease and immediacy of acquiring knowledge through these digital oracles. In this exploration, we delve into the sociocultural ramifications of LLMs, which, through their nuanced responses, may be subtly etching their own stereotypes, an algorithmic or AI monoculture. We hypothesize that the convergence of pre-training data, fine-tuning datasets, and analogous guardrails across models may have birthed a distinct cultural lens. We unpack this concept through a short experiment navigating children's storytelling, occupational-ethnic biases, and self-diagnosed annotations, to find that there exists strong cosine similarity (0.87) of biases across these models, suggesting a similar perspective of ethnic stereotypes in occupations. This paper invites a reimagining of LLMs' societal role, especially as the new information gatekeepers, advocating for a paradigm shift towards diversity-rich landscapes over unintended monocultures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06919v1">Automating Thematic Analysis: How LLMs Analyse Controversial Topics</a></div>
    <div class="paper-meta">
      📅 2024-05-11
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are promising analytical tools. They can augment human epistemic, cognitive and reasoning abilities, and support 'sensemaking', making sense of a complex environment or subject by analysing large volumes of data with a sensitivity to context and nuance absent in earlier text processing systems. This paper presents a pilot experiment that explores how LLMs can support thematic analysis of controversial topics. We compare how human researchers and two LLMs GPT-4 and Llama 2 categorise excerpts from media coverage of the controversial Australian Robodebt scandal. Our findings highlight intriguing overlaps and variances in thematic categorisation between human and machine agents, and suggest where LLMs can be effective in supporting forms of discourse and thematic analysis. We argue LLMs should be used to augment, and not replace human interpretation, and we add further methodological insights and reflections to existing research on the application of automation to qualitative research methods. We also introduce a novel card-based design toolkit, for both researchers and practitioners to further interrogate LLMs as analytical tools.
    </details>
</div>
