# llm - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00028v2">Synergizing LLM Agents and Knowledge Graph for Socioeconomic Prediction in LBSN</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      The fast development of location-based social networks (LBSNs) has led to significant changes in society, resulting in popular studies of using LBSN data for socioeconomic prediction, e.g., regional population and commercial activity estimation. Existing studies design various graphs to model heterogeneous LBSN data, and further apply graph representation learning methods for socioeconomic prediction. However, these approaches heavily rely on heuristic ideas and expertise to extract task-relevant knowledge from diverse data, which may not be optimal for specific tasks. Additionally, they tend to overlook the inherent relationships between different indicators, limiting the prediction accuracy. Motivated by the remarkable abilities of large language models (LLMs) in commonsense reasoning, embedding, and multi-agent collaboration, in this work, we synergize LLM agents and knowledge graph for socioeconomic prediction. We first construct a location-based knowledge graph (LBKG) to integrate multi-sourced LBSN data. Then we leverage the reasoning power of LLM agent to identify relevant meta-paths in the LBKG for each type of socioeconomic prediction task, and design a semantic-guided attention module for knowledge fusion with meta-paths. Moreover, we introduce a cross-task communication mechanism to further enhance performance by enabling knowledge sharing across tasks at both LLM agent and KG levels. On the one hand, the LLM agents for different tasks collaborate to generate more diverse and comprehensive meta-paths. On the other hand, the embeddings from different tasks are adaptively merged for better socioeconomic prediction. Experiments on two datasets demonstrate the effectiveness of the synergistic design between LLM and KG, providing insights for information sharing across socioeconomic prediction tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04793v2">Zero-shot LLM-guided Counterfactual Generation: A Case Study on NLP Model Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 Longer version of short paper accepted at IEEE BigData 2024 (Main Track)
    </div>
    <details class="paper-abstract">
      With the development and proliferation of large, complex, black-box models for solving many natural language processing (NLP) tasks, there is also an increasing necessity of methods to stress-test these models and provide some degree of interpretability or explainability. While counterfactual examples are useful in this regard, automated generation of counterfactuals is a data and resource intensive process. such methods depend on models such as pre-trained language models that are then fine-tuned on auxiliary, often task-specific datasets, that may be infeasible to build in practice, especially for new tasks and data domains. Therefore, in this work we explore the possibility of leveraging large language models (LLMs) for zero-shot counterfactual generation in order to stress-test NLP models. We propose a structured pipeline to facilitate this generation, and we hypothesize that the instruction-following and textual understanding capabilities of recent LLMs can be effectively leveraged for generating high quality counterfactuals in a zero-shot manner, without requiring any training or fine-tuning. Through comprehensive experiments on a variety of propreitary and open-source LLMs, along with various downstream tasks in NLP, we explore the efficacy of LLMs as zero-shot counterfactual generators in evaluating and explaining black-box NLP models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12395v1">Do LLMs Understand Ambiguity in Text? A Case Study in Open-world Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 Accepted at the REU Symposium at IEEE BigData 2024
    </div>
    <details class="paper-abstract">
      Ambiguity in natural language poses significant challenges to Large Language Models (LLMs) used for open-domain question answering. LLMs often struggle with the inherent uncertainties of human communication, leading to misinterpretations, miscommunications, hallucinations, and biased responses. This significantly weakens their ability to be used for tasks like fact-checking, question answering, feature extraction, and sentiment analysis. Using open-domain question answering as a test case, we compare off-the-shelf and few-shot LLM performance, focusing on measuring the impact of explicit disambiguation strategies. We demonstrate how simple, training-free, token-level disambiguation methods may be effectively used to improve LLM performance for ambiguous question answering tasks. We empirically show our findings and discuss best practices and broader impacts regarding ambiguity in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19572v4">ChunkRAG: Novel LLM-Chunk Filtering Method for RAG Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems using large language models (LLMs) often generate inaccurate responses due to the retrieval of irrelevant or loosely related information. Existing methods, which operate at the document level, fail to effectively filter out such content. We propose LLM-driven chunk filtering, ChunkRAG, a framework that enhances RAG systems by evaluating and filtering retrieved information at the chunk level. Our approach employs semantic chunking to divide documents into coherent sections and utilizes LLM-based relevance scoring to assess each chunk's alignment with the user's query. By filtering out less pertinent chunks before the generation phase, we significantly reduce hallucinations and improve factual accuracy. Experiments show that our method outperforms existing RAG models, achieving higher accuracy on tasks requiring precise information retrieval. This advancement enhances the reliability of RAG systems, making them particularly beneficial for applications like fact-checking and multi-hop reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12355v1">DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The challenge in LLM-based video understanding lies in preserving visual and semantic information in long videos while maintaining a memory-affordable token count. However, redundancy and correspondence in videos have hindered the performance potential of existing methods. Through statistical learning on current datasets, we observe that redundancy occurs in both repeated and answer-irrelevant frames, and the corresponding frames vary with different questions. This suggests the possibility of adopting dynamic encoding to balance detailed video information preservation with token budget reduction. To this end, we propose a dynamic cooperative network, DynFocus, for memory-efficient video encoding in this paper. Specifically, i) a Dynamic Event Prototype Estimation (DPE) module to dynamically select meaningful frames for question answering; (ii) a Compact Cooperative Encoding (CCE) module that encodes meaningful frames with detailed visual appearance and the remaining frames with sketchy perception separately. We evaluate our method on five publicly available benchmarks, and experimental results consistently demonstrate that our method achieves competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05085v2">Multi-Head RAG: Solving Multi-Aspect Problems with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) enhances the abilities of Large Language Models (LLMs) by enabling the retrieval of documents into the LLM context to provide more accurate and relevant responses. Existing RAG solutions do not focus on queries that may require fetching multiple documents with substantially different contents. Such queries occur frequently, but are challenging because the embeddings of these documents may be distant in the embedding space, making it hard to retrieve them all. This paper introduces Multi-Head RAG (MRAG), a novel scheme designed to address this gap with a simple yet powerful idea: leveraging activations of Transformer's multi-head attention layer, instead of the decoder layer, as keys for fetching multi-aspect documents. The driving motivation is that different attention heads can learn to capture different data aspects. Harnessing the corresponding activations results in embeddings that represent various facets of data items and queries, improving the retrieval accuracy for complex queries. We provide an evaluation methodology and metrics, multi-aspect datasets that we release online, and real-world use cases to demonstrate MRAG's effectiveness, showing improvements of up to 20% in relevance over standard RAG baselines. MRAG can be seamlessly integrated with existing RAG frameworks and benchmarking tools like RAGAS as well as different classes of data stores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12307v1">Balancing Accuracy and Efficiency in Multi-Turn Intent Classification for LLM-Powered Dialog Systems in Production</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      Accurate multi-turn intent classification is essential for advancing conversational AI systems. However, challenges such as the scarcity of comprehensive datasets and the complexity of contextual dependencies across dialogue turns hinder progress. This paper presents two novel approaches leveraging Large Language Models (LLMs) to enhance scalability and reduce latency in production dialogue systems. First, we introduce Symbol Tuning, which simplifies intent labels to reduce task complexity and improve performance in multi-turn dialogues. Second, we propose C-LARA (Consistency-aware, Linguistics Adaptive Retrieval Augmentation), a framework that employs LLMs for data augmentation and pseudo-labeling to generate synthetic multi-turn dialogues. These enriched datasets are used to fine-tune a small, efficient model suitable for deployment. Experiments conducted on multilingual dialogue datasets demonstrate significant improvements in classification accuracy and resource efficiency. Our methods enhance multi-turn intent classification accuracy by 5.09%, reduce annotation costs by 40%, and enable scalable deployment in low-resource multilingual industrial systems, highlighting their practicality and impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15000v3">Divide-or-Conquer? Which Part Should You Distill Your LLM?</a></div>
    <div class="paper-meta">
      📅 2024-11-19
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2024
    </div>
    <details class="paper-abstract">
      Recent methods have demonstrated that Large Language Models (LLMs) can solve reasoning tasks better when they are encouraged to solve subtasks of the main task first. In this paper we devise a similar strategy that breaks down reasoning tasks into a problem decomposition phase and a problem solving phase and show that the strategy is able to outperform a single stage solution. Further, we hypothesize that the decomposition should be easier to distill into a smaller model compared to the problem solving because the latter requires large amounts of domain knowledge while the former only requires learning general problem solving strategies. We propose methods to distill these two capabilities and evaluate their impact on reasoning outcomes and inference cost. We find that we can distill the problem decomposition phase and at the same time achieve good generalization across tasks, datasets, and models. However, it is harder to distill the problem solving capability without losing performance and the resulting distilled model struggles with generalization. These results indicate that by using smaller, distilled problem decomposition models in combination with problem solving LLMs we can achieve reasoning with cost-efficient inference and local adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06145v2">Escalating LLM-based Code Translation Benchmarking into the Class-level Era</a></div>
    <div class="paper-meta">
      📅 2024-11-19
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have significantly improved automated code translation, often achieving over 80% accuracy on existing benchmarks. However, most of these benchmarks consist of short, standalone, algorithmic samples that do not reflect practical coding tasks. To address this gap, we introduce ClassEval-T, a class-level code translation benchmark designed to assess LLM performance on real-world coding scenarios. Built upon ClassEval, a class-level Python code generation benchmark covering topics such as database operations and game design, ClassEval-T extends into Java and C++ with complete code samples and test suites, requiring 360 person-hours for manual migration. We propose three translation strategies (holistic, min-dependency, and standalone) and evaluate six recent LLMs across various families and sizes on ClassEval-T. Results reveal a significant performance drop compared to method-level benchmarks, highlighting discrepancies among LLMs and demonstrating ClassEval-T's effectiveness. We further analyze LLMs' dependency awareness in translating class samples and categorize 1,397 failure cases by the best-performing LLM for practical insights and future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05981v4">ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance on language tasks but face challenges when deployed on resource-constrained devices due to their extensive parameters and reliance on dense multiplications, resulting in high memory demands and latency bottlenecks. Shift-and-add reparameterization offers a promising solution by replacing costly multiplications with hardware-friendly primitives in both the attention and multi-layer perceptron (MLP) layers of an LLM. However, current reparameterization techniques require training from scratch or full parameter fine-tuning to restore accuracy, which is resource-intensive for LLMs. To address this, we propose accelerating pretrained LLMs through post-training shift-and-add reparameterization, creating efficient multiplication-free models, dubbed ShiftAddLLM. Specifically, we quantize each weight matrix into binary matrices paired with group-wise scaling factors. The associated multiplications are reparameterized into (1) shifts between activations and scaling factors and (2) queries and adds according to the binary matrices. To reduce accuracy loss, we present a multi-objective optimization method to minimize both weight and output activation reparameterization errors. Additionally, based on varying sensitivity across layers to reparameterization, we develop an automated bit allocation strategy to further reduce memory usage and latency. Experiments on five LLM families and eight tasks consistently validate the effectiveness of ShiftAddLLM, achieving average perplexity improvements of 5.6 and 22.7 points at comparable or lower latency compared to the most competitive quantized LLMs at 3 and 2 bits, respectively, and more than 80% memory and energy reductions over the original LLMs. Codes and models are available at https://github.com/GATECH-EIC/ShiftAddLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14469v1">Popular LLMs Amplify Race and Gender Disparities in Human Mobility</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied in areas influencing societal outcomes, it is critical to understand their tendency to perpetuate and amplify biases. This study investigates whether LLMs exhibit biases in predicting human mobility -- a fundamental human behavior -- based on race and gender. Using three prominent LLMs -- GPT-4, Gemini, and Claude -- we analyzed their predictions of visitations to points of interest (POIs) for individuals, relying on prompts that included names with and without explicit demographic details. We find that LLMs frequently reflect and amplify existing societal biases. Specifically, predictions for minority groups were disproportionately skewed, with these individuals being significantly less likely to be associated with wealth-related points of interest (POIs). Gender biases were also evident, as female individuals were consistently linked to fewer career-related POIs compared to their male counterparts. These biased associations suggest that LLMs not only mirror but also exacerbate societal stereotypes, particularly in contexts involving race and gender.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11984v1">Understanding Chain-of-Thought in LLMs through Information Theory</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in complex reasoning tasks through Chain-of-Thought (CoT) reasoning, allowing models to break down problems into manageable sub-tasks. However, existing CoT evaluation techniques either require annotated CoT data or fall short in accurately assessing intermediate reasoning steps, leading to high rates of false positives. In this paper, we formalize CoT reasoning in LLMs through an information-theoretic lens. Specifically, our framework quantifies the `information gain' at each reasoning step, enabling the identification of failure modes in LLMs without the need for expensive annotated datasets. We demonstrate the efficacy of our approach through extensive experiments on toy and GSM-8K data, where it significantly outperforms existing outcome-based methods by providing more accurate insights into model performance on individual tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07681v2">What Do Learning Dynamics Reveal About Generalization in LLM Reasoning?</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Despite the remarkable capabilities of modern large language models (LLMs), the mechanisms behind their problem-solving abilities remain elusive. In this work, we aim to better understand how the learning dynamics of LLM finetuning shapes downstream generalization. Our analysis focuses on reasoning tasks, whose problem structure allows us to distinguish between memorization (the exact replication of reasoning steps from the training data) and performance (the correctness of the final solution). We find that a model's generalization behavior can be effectively characterized by a training metric we call pre-memorization train accuracy: the accuracy of model samples on training queries before they begin to copy the exact reasoning steps from the training set. On the dataset level, this metric is able to reliably predict test accuracy, achieving $R^2$ of around or exceeding 0.9 across various models (Llama3 8, Gemma2 9B), datasets (GSM8k, MATH), and training configurations. On a per-example level, this metric is also indicative of whether individual model predictions are robust to perturbations in the training query. By connecting a model's learning behavior to its generalization, pre-memorization train accuracy can guide targeted improvements to training strategies. We focus on data curation as an example, and show that prioritizing examples with low pre-memorization accuracy leads to 1.5-2x improvements in data efficiency compared to i.i.d. data scaling, and outperforms other standard data curation techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11829v1">Tackling prediction tasks in relational databases with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Though large language models (LLMs) have demonstrated exceptional performance across numerous problems, their application to predictive tasks in relational databases remains largely unexplored. In this work, we address the notion that LLMs cannot yield satisfactory results on relational databases due to their interconnected tables, complex relationships, and heterogeneous data types. Using the recently introduced RelBench benchmark, we demonstrate that even a straightforward application of LLMs achieves competitive performance on these tasks. These findings establish LLMs as a promising new baseline for ML on relational databases and encourage further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11752v1">sMoRe: Enhancing Object Manipulation and Organization in Mixed Reality Spaces with LLMs and Generative AI</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      In mixed reality (MR) environments, understanding space and creating virtual objects is crucial to providing an intuitive and rich user experience. This paper introduces sMoRe (Spatial Mapping and Object Rendering Environment), an MR application that combines Generative AI (GenAI) with large language models (LLMs) to assist users in creating, placing, and managing virtual objects within physical spaces. sMoRe allows users to use voice or typed text commands to create and place virtual objects using GenAI while specifying spatial constraints. The system leverages LLMs to interpret users' commands, analyze the current scene, and identify optimal locations. Additionally, sMoRe integrates text-to-3D generative AI to dynamically create 3D objects based on users' descriptions. Our user study demonstrates the effectiveness of sMoRe in enhancing user comprehension, interaction, and organization of the MR environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06153v2">AgentSquare: Automatic LLM Agent Search in Modular Design Space</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to a rapid growth of agentic systems capable of handling a wide range of complex tasks. However, current research largely relies on manual, task-specific design, limiting their adaptability to novel tasks. In this paper, we introduce a new research problem: Modularized LLM Agent Search (MoLAS). We propose a modular design space that abstracts existing LLM agent designs into four fundamental modules with uniform IO interface: Planning, Reasoning, Tool Use, and Memory. Building on this design space, we present a novel LLM agent search framework called AgentSquare, which introduces two core mechanisms, i.e., module evolution and recombination, to efficiently search for optimized LLM agents. To further accelerate the process, we design a performance predictor that uses in-context surrogate models to skip unpromising agent designs. Extensive experiments across six benchmarks, covering the diverse scenarios of web, embodied, tool use and game applications, show that AgentSquare substantially outperforms hand-crafted agents, achieving an average performance gain of 17.2% against best-known human designs. Moreover, AgentSquare can generate interpretable design insights, enabling a deeper understanding of agentic architecture and its impact on task performance. We believe that the modular design space and AgentSquare search framework offer a platform for fully exploiting the potential of prior successful designs and consolidating the collective efforts of research community. Code repo is available at https://github.com/tsinghua-fib-lab/AgentSquare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11745v1">BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 HPCA 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across various machine learning tasks. Yet the substantial memory footprint of LLMs significantly hinders their deployment. In this paper, we improve the accessibility of LLMs through BitMoD, an algorithm-hardware co-design solution that enables efficient LLM acceleration at low weight precision. On the algorithm side, BitMoD introduces fine-grained data type adaptation that uses a different numerical data type to quantize a group of (e.g., 128) weights. Through the careful design of these new data types, BitMoD is able to quantize LLM weights to very low precision (e.g., 4 bits and 3 bits) while maintaining high accuracy. On the hardware side, BitMoD employs a bit-serial processing element to easily support multiple numerical precisions and data types; our hardware design includes two key innovations: First, it employs a unified representation to process different weight data types, thus reducing the hardware cost. Second, it adopts a bit-serial dequantization unit to rescale the per-group partial sum with minimal hardware overhead. Our evaluation on six representative LLMs demonstrates that BitMoD significantly outperforms state-of-the-art LLM quantization and acceleration methods. For discriminative tasks, BitMoD can quantize LLM weights to 4-bit with $<\!0.5\%$ accuracy loss on average. For generative tasks, BitMoD is able to quantize LLM weights to 3-bit while achieving better perplexity than prior LLM quantization scheme. Combining the superior model performance with an efficient accelerator design, BitMoD achieves an average of $1.69\times$ and $1.48\times$ speedups compared to prior LLM accelerators ANT and OliVe, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11582v1">Exploring LLMs for Verifying Technical System Specifications Against Requirements</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Submitted to 3rd IEEE Industrial Electronics Society Annual Online Conference (ONCON)
    </div>
    <details class="paper-abstract">
      Requirements engineering is a knowledge intensive process and crucial for the success of engineering projects. The field of knowledge-based requirements engineering (KBRE) aims to support engineers by providing knowledge to assist in the elicitation, validation, and management of system requirements. The advent of large language models (LLMs) opens new opportunities in the field of KBRE. This work experimentally investigates the potential of LLMs in requirements verification. Therein, LLMs are provided with a set of requirements and a textual system specification and are prompted to assess which requirements are fulfilled by the system specification. Different experimental variables such as system specification complexity, the number of requirements, and prompting strategies were analyzed. Formal rule-based systems serve as a benchmark to compare LLM performance to. Requirements and system specifications are derived from the smart-grid domain. Results show that advanced LLMs, like GPT-4o and Claude 3.5 Sonnet, achieved f1-scores between 79 % and 94 % in identifying non-fulfilled requirements, indicating potential for LLMs to be leveraged for requirements verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11560v1">Topology-aware Preemptive Scheduling for Co-located LLM Workloads</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 17 Pages, 11 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Hosting diverse large language model workloads in a unified resource pool through co-location is cost-effective. For example, long-running chat services generally follow diurnal traffic patterns, which inspire co-location of batch jobs to fulfill resource valleys between successive peaks, and thus to saturate resource allocation in cluster-wide scope. These heterogeneous workloads often have different business priorities, and therefore preemption can be leveraged for resource elasticity. However, workloads often have distinct topology preferences as well. The resources released by lower-priority instances may fail to meet the requirements of high-priority online services which are usually latency-sensitive. The root cause behind such mis-match is a lack of topology awareness of resource scheduler, especially during preemption. To bridge this gap, we develop a fine-grained topology-aware method for preemptive scheduling of hybrid workloads. The method ensures that the resources freed by preempted tasks adhere to the topological affinity needs of high-priority preemptors in a guaranteed or best-effort manner. This dynamic alignment significantly increases the efficiency of preemption and improves overall scheduled performance for LLM workloads by $55\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16937v2">A Complete Survey on LLM-based AI Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 23 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The past few decades have witnessed an upsurge in data, forming the foundation for data-hungry, learning-based AI technology. Conversational agents, often referred to as AI chatbots, rely heavily on such data to train large language models (LLMs) and generate new content (knowledge) in response to user prompts. With the advent of OpenAI's ChatGPT, LLM-based chatbots have set new standards in the AI community. This paper presents a complete survey of the evolution and deployment of LLM-based chatbots in various sectors. We first summarize the development of foundational chatbots, followed by the evolution of LLMs, and then provide an overview of LLM-based chatbots currently in use and those in the development phase. Recognizing AI chatbots as tools for generating new knowledge, we explore their diverse applications across various industries. We then discuss the open challenges, considering how the data used to train the LLMs and the misuse of the generated knowledge can cause several issues. Finally, we explore the future outlook to augment their efficiency and reliability in numerous applications. By addressing key milestones and the present-day context of LLM-based chatbots, our survey invites readers to delve deeper into this realm, reflecting on how their next generation will reshape conversational AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11521v1">Preempting Text Sanitization Utility in Resource-Constrained Privacy-Preserving LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Individuals have been increasingly interacting with online Large Language Models (LLMs), both in their work and personal lives. These interactions raise privacy issues as the LLMs are typically hosted by third-parties who can gather a variety of sensitive information about users and their companies. Text Sanitization techniques have been proposed in the literature and can be used to sanitize user prompts before sending them to the LLM. However, sanitization has an impact on the downstream task performed by the LLM, and often to such an extent that it leads to unacceptable results for the user. This is not just a minor annoyance, with clear monetary consequences as LLM services charge on a per use basis as well as great amount of computing resources wasted. We propose an architecture leveraging a Small Language Model (SLM) at the user-side to help estimate the impact of sanitization on a prompt before it is sent to the LLM, thus preventing resource losses. Our evaluation of this architecture revealed a significant problem with text sanitization based on Differential Privacy, on which we want to draw the attention of the community for further investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.12737v2">LLM App Store Analysis: A Vision and Roadmap</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      The rapid growth and popularity of large language model (LLM) app stores have created new opportunities and challenges for researchers, developers, users, and app store managers. As the LLM app ecosystem continues to evolve, it is crucial to understand the current landscape and identify potential areas for future research and development. This paper presents a forward-looking analysis of LLM app stores, focusing on key aspects such as data mining, security risk identification, development assistance, and market dynamics. Our comprehensive examination extends to the intricate relationships between various stakeholders and the technological advancements driving the ecosystem's growth. We explore the ethical considerations and potential societal impacts of widespread LLM app adoption, highlighting the need for responsible innovation and governance frameworks. By examining these aspects, we aim to provide a vision for future research directions and highlight the importance of collaboration among stakeholders to address the challenges and opportunities within the LLM app ecosystem. The insights and recommendations provided in this paper serve as a foundation for driving innovation, ensuring responsible development, and creating a thriving, user-centric LLM app landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18492v3">LLMs and Memorization: On Quality and Specificity of Copyright Compliance</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 10 pages, 3 figures, AIES 2024 conference
    </div>
    <details class="paper-abstract">
      Memorization in large language models (LLMs) is a growing concern. LLMs have been shown to easily reproduce parts of their training data, including copyrighted work. This is an important problem to solve, as it may violate existing copyright laws as well as the European AI Act. In this work, we propose a systematic analysis to quantify the extent of potential copyright infringements in LLMs using European law as an example. Unlike previous work, we evaluate instruction-finetuned models in a realistic end-user scenario. Our analysis builds on a proposed threshold of 160 characters, which we borrow from the German Copyright Service Provider Act and a fuzzy text matching algorithm to identify potentially copyright-infringing textual reproductions. The specificity of countermeasures against copyright infringement is analyzed by comparing model behavior on copyrighted and public domain data. We investigate what behaviors models show instead of producing protected text (such as refusal or hallucination) and provide a first legal assessment of these behaviors. We find that there are huge differences in copyright compliance, specificity, and appropriate refusal among popular LLMs. Alpaca, GPT 4, GPT 3.5, and Luminous perform best in our comparison, with OpenGPT-X, Alpaca, and Luminous producing a particularly low absolute number of potential copyright violations. Code can be found at https://github.com/felixbmuller/llms-memorization-copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20911v2">Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 v0.2 (evaluated on more agents)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10370v2">Grounded 3D-LLM with Referent Tokens</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Prior studies on 3D scene understanding have primarily developed specialized models for specific tasks or required task-specific fine-tuning. In this study, we propose Grounded 3D-LLM, which explores the potential of 3D large multi-modal models (3D LMMs) to consolidate various 3D vision tasks within a unified generative framework. The model uses scene referent tokens as special noun phrases to reference 3D scenes, enabling it to handle sequences that interleave 3D and textual data. Per-task instruction-following templates are employed to ensure natural and diversity in translating 3D vision tasks into language formats. To facilitate the use of referent tokens in subsequent language modeling, we provide a large-scale, automatically curated grounded scene-text dataset with over 1 million phrase-to-region correspondences and introduce Contrastive Language-Scene Pre-training (CLASP) to perform phrase-level scene-text alignment using this data. Our comprehensive evaluation covers open-ended tasks like dense captioning and 3D question answering, alongside close-ended tasks such as object detection and language grounding. Experiments across multiple 3D benchmarks reveal the leading performance and the broad applicability of Grounded 3D-LLM. Code and datasets are available at the https://groundedscenellm.github.io/grounded_3d-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19979v3">Enhancing High-order Interaction Awareness in LLM-based Recommender Model</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Long paper accepted to EMNLP 2024 Main. 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated prominent reasoning capabilities in recommendation tasks by transforming them into text-generation tasks. However, existing approaches either disregard or ineffectively model the user-item high-order interactions. To this end, this paper presents an enhanced LLM-based recommender (ELMRec). We enhance whole-word embeddings to substantially enhance LLMs' interpretation of graph-constructed interactions for recommendations, without requiring graph pre-training. This finding may inspire endeavors to incorporate rich knowledge graphs into LLM-based recommenders via whole-word embedding. We also found that LLMs often recommend items based on users' earlier interactions rather than recent ones, and present a reranking solution. Our ELMRec outperforms state-of-the-art (SOTA) methods in both direct and sequential recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03578v1">PerfCodeGen: Improving Performance of LLM Generated Code with Execution Feedback</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for assisting in software development tasks, yet their performance evaluations have narrowly focused on the functional correctness of generated code. Human programmers, however, require LLM-generated code to be not only correct but also optimally efficient. We propose PerfCodeGen, a training-free framework that enhances the performance of LLM-generated code by incorporating feedback based on runtime during test case execution into the self-refinement iterations. With PerfCodeGen, we achieve speedups for a significantly higher proportion of problems compared to using the base LLM with sophisticated prompting techniques. Applied to open language models like Phi-3-mini, PerfCodeGen achieves runtime efficiency comparable to prompting powerful closed models like GPT-4. We achieve state-of-the-art runtime efficiency on benchmarks such as HumanEval, MBPP, and APPS, frequently surpassing the ground truth reference solutions with PerfCodeGen using GPT-3.5 and GPT-4. Additionally, we demonstrate the effectiveness of our approach in enhancing code quality across a range of open LLMs of varying sizes including Phi-3-mini, Llama 3 8B, Mixtral 8x7B, Command R, and Llama 3 70B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08527v2">Optimized Feature Generation for Tabular Data via LLMs with Decision Tree Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      In tabular prediction tasks, tree-based models combined with automated feature engineering methods often outperform deep learning approaches that rely on learned representations. While these feature engineering techniques are effective, they typically depend on a pre-defined search space and primarily use validation scores for feature selection, thereby missing valuable insights from previous experiments. To address these limitations, we propose a novel tabular learning framework that utilizes large language models (LLMs), termed Optimizing Column feature generator with decision Tree reasoning (OCTree). Our key idea is to leverage the reasoning capabilities of LLMs to identify effective feature generation rules without manually specifying the search space and provide language-based reasoning information highlighting past experiments as feedback for iterative rule improvements. We use decision trees to convey this reasoning information, as they can be easily represented in natural language, effectively providing knowledge from prior experiments (i.e., the impact of the generated features on performance) to the LLMs. Our empirical results demonstrate that OCTree consistently enhances the performance of various prediction models across diverse benchmarks, outperforming competing automated feature engineering methods. Code is available at https://github.com/jaehyun513/OCTree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11295v1">Transcending Language Boundaries: Harnessing LLMs for Low-Resource Language Translation</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across a wide range of tasks and domains. However, their performance in low-resource language translation, particularly when translating into these languages, remains underexplored. This gap poses significant challenges, as linguistic barriers hinder the cultural preservation and development of minority communities. To address this issue, this paper introduces a novel retrieval-based method that enhances translation quality for low-resource languages by focusing on key terms, which involves translating keywords and retrieving corresponding examples from existing data. To evaluate the effectiveness of this method, we conducted experiments translating from English into three low-resource languages: Cherokee, a critically endangered indigenous language of North America; Tibetan, a historically and culturally significant language in Asia; and Manchu, a language with few remaining speakers. Our comparison with the zero-shot performance of GPT-4o and LLaMA 3.1 405B, highlights the significant challenges these models face when translating into low-resource languages. In contrast, our retrieval-based method shows promise in improving both word-level accuracy and overall semantic understanding by leveraging existing resources more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03816v3">ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search</a></div>
    <div class="paper-meta">
      📅 2024-11-18
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Recent methodologies in LLM self-training mostly rely on LLM generating responses and filtering those with correct output answers as training data. This approach often yields a low-quality fine-tuning training set (e.g., incorrect plans or intermediate reasoning). In this paper, we develop a reinforced self-training approach, called ReST-MCTS*, based on integrating process reward guidance with tree search MCTS* for collecting higher-quality reasoning traces as well as per-step value to train policy and reward models. ReST-MCTS* circumvents the per-step manual annotation typically used to train process rewards by tree-search-based reinforcement learning: Given oracle final correct answers, ReST-MCTS* is able to infer the correct process rewards by estimating the probability this step can help lead to the correct answer. These inferred rewards serve dual purposes: they act as value targets for further refining the process reward model and also facilitate the selection of high-quality traces for policy model self-training. We first show that the tree-search policy in ReST-MCTS* achieves higher accuracy compared with prior LLM reasoning baselines such as Best-of-N and Tree-of-Thought, within the same search budget. We then show that by using traces searched by this tree-search policy as training data, we can continuously enhance the three language models for multiple iterations, and outperform other self-training algorithms such as ReST$^\text{EM}$ and Self-Rewarding LM. We release all code at https://github.com/THUDM/ReST-MCTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.14782v2">Understanding the Role of Textual Prompts in LLM for Time Series Forecasting: an Adapter View</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      In the burgeoning domain of Large Language Models (LLMs), there is a growing interest in applying LLM to time series forecasting, with multiple studies focused on leveraging textual prompts to further enhance the predictive prowess. This study aims to understand how and why the integration of textual prompts into LLM can effectively improve the prediction accuracy of time series, which is not obvious at the glance, given the significant domain gap between texts and time series. Our extensive examination leads us to believe that (a) adding text prompts is roughly equivalent to introducing additional adapters, and (b) It is the introduction of learnable parameters rather than textual information that aligns the LLM with the time series forecasting task, ultimately enhancing prediction accuracy. Inspired by this discovery, we developed four adapters that explicitly address the gap between LLM and time series, and further improve the prediction accuracy. Overall,our work highlights how textual prompts enhance LLM accuracy in time series forecasting and suggests new avenues for continually improving LLM-based time series analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11285v1">Zero-Shot Automatic Annotation and Instance Segmentation using LLM-Generated Datasets: Eliminating Field Imaging and Manual Annotation for Deep Learning Model Development</a></div>
    <div class="paper-meta">
      📅 2024-11-18
    </div>
    <details class="paper-abstract">
      Currently, deep learning-based instance segmentation for various applications (e.g., Agriculture) is predominantly performed using a labor-intensive process involving extensive field data collection using sophisticated sensors, followed by careful manual annotation of images, presenting significant logistical and financial challenges to researchers and organizations. The process also slows down the model development and training process. In this study, we presented a novel method for deep learning-based instance segmentation of apples in commercial orchards that eliminates the need for labor-intensive field data collection and manual annotation. Utilizing a Large Language Model (LLM), we synthetically generated orchard images and automatically annotated them using the Segment Anything Model (SAM) integrated with a YOLO11 base model. This method significantly reduces reliance on physical sensors and manual data processing, presenting a major advancement in "Agricultural AI". The synthetic, auto-annotated dataset was used to train the YOLO11 model for Apple instance segmentation, which was then validated on real orchard images. The results showed that the automatically generated annotations achieved a Dice Coefficient of 0.9513 and an IoU of 0.9303, validating the accuracy and overlap of the mask annotations. All YOLO11 configurations, trained solely on these synthetic datasets with automated annotations, accurately recognized and delineated apples, highlighting the method's efficacy. Specifically, the YOLO11m-seg configuration achieved a mask precision of 0.902 and a mask mAP@50 of 0.833 on test images collected from a commercial orchard. Additionally, the YOLO11l-seg configuration outperformed other models in validation on 40 LLM-generated images, achieving the highest mask precision and mAP@50 metrics. Keywords: YOLO, SAM, SAMv2, YOLO11, YOLOv11, Segment Anything, YOLO-SAM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12764v1">SEFD: Semantic-Enhanced Framework for Detecting LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2024-11-17
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) has created an urgent need for robust tools to detect LLM-generated text, especially in light of \textit{paraphrasing} techniques that often evade existing detection methods. To address this challenge, we present a novel semantic-enhanced framework for detecting LLM-generated text (SEFD) that leverages a retrieval-based mechanism to fully utilize text semantics. Our framework improves upon existing detection methods by systematically integrating retrieval-based techniques with traditional detectors, employing a carefully curated retrieval mechanism that strikes a balance between comprehensive coverage and computational efficiency. We showcase the effectiveness of our approach in sequential text scenarios common in real-world applications, such as online forums and Q\&A platforms. Through comprehensive experiments across various LLM-generated texts and detection methods, we demonstrate that our framework substantially enhances detection accuracy in paraphrasing scenarios while maintaining robustness for standard LLM-generated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19336v3">Improving LLM Classification of Logical Errors by Integrating Error Relationship into Prompts</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 Published in ITS 2024 (Best Paper Award)
    </div>
    <details class="paper-abstract">
      LLMs trained in the understanding of programming syntax are now providing effective assistance to developers and are being used in programming education such as in generation of coding problem examples or providing code explanations. A key aspect of programming education is understanding and dealing with error message. However, 'logical errors' in which the program operates against the programmer's intentions do not receive error messages from the compiler. In this study, building on existing research on programming errors, we first define the types of logical errors that can occur in programming in general. Based on the definition, we propose an effective approach for detecting logical errors with LLMs that makes use of relations among error types in the Chain-of-Thought and Tree-of-Thought prompts. The experimental results indicate that when such logical error descriptions in the prompt are used, the average classifition performance is about 21% higher than the ones without them. We also conducted an experiment for exploiting the relations among errors in generating a new logical error dataset using LLMs. As there is very limited dataset for logical errors such benchmark dataset can be very useful for various programming related applications. We expect that our work can assist novice programmers in identifying the causes of code errors and correct them more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.20098v2">Web2Code: A Large-scale Webpage-to-Code Dataset and Evaluation Framework for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 NeurIPS 2024 Datasets and Benchmarks Camera-ready Version. Website at https://mbzuai-llm.github.io/webpage2code/
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown impressive success across modalities such as image, video, and audio in a variety of understanding and generation tasks. However, current MLLMs are surprisingly poor at understanding webpage screenshots and generating their corresponding HTML code. To address this problem, we propose $\texttt{Web2Code}$, a benchmark consisting of a new large-scale webpage-to-code dataset for instruction tuning and an evaluation framework for the webpage understanding and HTML code translation abilities of MLLMs. For dataset construction, we leverage pretrained LLMs to enhance existing webpage-to-code datasets as well as generate a diverse pool of new webpages rendered into images. Specifically, the inputs are webpage images and instructions, while the responses are the webpage's HTML code. We further include diverse natural language QA pairs about the webpage content in the responses to enable a more comprehensive understanding of the web content. To evaluate model performance in these tasks, we develop an evaluation framework for testing MLLMs' abilities in webpage understanding and web-to-code generation. Extensive experiments show that our proposed dataset is beneficial not only to our proposed tasks but also in the general visual domain. We hope our work will contribute to the development of general MLLMs suitable for web-based content generation and task automation. Our data and code are available at https://github.com/MBZUAI-LLM/web2code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18528v2">PrExMe! Large Scale Prompt Exploration of Open Source LLMs for Machine Translation and Summarization Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 EMNLP 2024 main; camera-ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized NLP research. Notably, in-context learning enables their use as evaluation metrics for natural language generation, making them particularly advantageous in low-resource scenarios and time-restricted applications. In this work, we introduce PrExMe, a large-scale Prompt Exploration for Metrics, where we evaluate more than 720 prompt templates for open-source LLM-based metrics on machine translation (MT) and summarization datasets, totalling over 6.6M evaluations. This extensive comparison (1) benchmarks recent open-source LLMs as metrics and (2) explores the stability and variability of different prompting strategies. We discover that, on the one hand, there are scenarios for which prompts are stable. For instance, some LLMs show idiosyncratic preferences and favor to grade generated texts with textual labels while others prefer to return numeric scores. On the other hand, the stability of prompts and model rankings can be susceptible to seemingly innocuous changes. For example, changing the requested output format from "0 to 100" to "-1 to +1" can strongly affect the rankings in our evaluation. Our study contributes to understanding the impact of different prompting approaches on LLM-based metrics for MT and summarization evaluation, highlighting the most stable prompting patterns and potential limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01837v1">Enabling Explainable Recommendation in E-commerce with LLM-powered Product Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 This paper was accepted by The First International OpenKG Workshop Large Knowledge-Enhanced Models @IJCAI 2024
    </div>
    <details class="paper-abstract">
      How to leverage large language model's superior capability in e-commerce recommendation has been a hot topic. In this paper, we propose LLM-PKG, an efficient approach that distills the knowledge of LLMs into product knowledge graph (PKG) and then applies PKG to provide explainable recommendations. Specifically, we first build PKG by feeding curated prompts to LLM, and then map LLM response to real enterprise products. To mitigate the risks associated with LLM hallucination, we employ rigorous evaluation and pruning methods to ensure the reliability and availability of the KG. Through an A/B test conducted on an e-commerce website, we demonstrate the effectiveness of LLM-PKG in driving user engagements and transactions significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05365v4">FiSTECH: Financial Style Transfer to Enhance Creativity without Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 10 pages, 14 figures, 5 tables, conference
    </div>
    <details class="paper-abstract">
      Recent trends in Generative AI have emerged towards fine-tuning foundational large language models (LLMs) to create domain-specific LLMs for automation and chatbot-like applications. Specialized applications for analytics-heavy domains such as Financial report generation require specific writing styles that comprise compound and creative sentences with minimized hallucinations. In this work, we explore the self-corrective auto-regressive qualities of LLMs to learn creativity in writing styles with minimal prompting. We propose a novel two-stage fine-tuning (FT) strategy wherein in the first stage public domain financial reports are used to train for writing styles while allowing the LLM to hallucinate. In the second stage the examples of hallucinations are manually corrected and further used to fine-tune the LLM. The finally trained LLM learns to generate specific financial report sections using minimal instructions and tabular data inputs while ensuring low fine-tuning costs. Our proposed two-stage fine-tuning boosts the accuracy of financial questions answering by two-folds while reducing hallucinations by over 50%. Also, the fine-tuned model has lower perplexity, improved ROUGE, TER and BLEU scores, higher creativity and knowledge density with lower uncertainty and cross entropy than base LLMs. Thus, the proposed framework can be generalized to train creativity in LLMs by first allowing them to hallucinate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10954v1">Dialectal Toxicity Detection: Evaluating LLM-as-a-Judge Consistency Across Language Varieties</a></div>
    <div class="paper-meta">
      📅 2024-11-17
    </div>
    <details class="paper-abstract">
      There has been little systematic study on how dialectal differences affect toxicity detection by modern LLMs. Furthermore, although using LLMs as evaluators ("LLM-as-a-judge") is a growing research area, their sensitivity to dialectal nuances is still underexplored and requires more focused attention. In this paper, we address these gaps through a comprehensive toxicity evaluation of LLMs across diverse dialects. We create a multi-dialect dataset through synthetic transformations and human-assisted translations, covering 10 language clusters and 60 varieties. We then evaluated three LLMs on their ability to assess toxicity across multilingual, dialectal, and LLM-human consistency. Our findings show that LLMs are sensitive in handling both multilingual and dialectal variations. However, if we have to rank the consistency, the weakest area is LLM-human agreement, followed by dialectal consistency. Code repository: \url{https://github.com/ffaisal93/dialect_toxicity_llm_judge}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10937v1">Memory-Augmented Multimodal LLMs for Surgical VQA via Self-Contained Inquiry</a></div>
    <div class="paper-meta">
      📅 2024-11-17
    </div>
    <details class="paper-abstract">
      Comprehensively understanding surgical scenes in Surgical Visual Question Answering (Surgical VQA) requires reasoning over multiple objects. Previous approaches address this task using cross-modal fusion strategies to enhance reasoning ability. However, these methods often struggle with limited scene understanding and question comprehension, and some rely on external resources (e.g., pre-extracted object features), which can introduce errors and generalize poorly across diverse surgical environments. To address these challenges, we propose SCAN, a simple yet effective memory-augmented framework that leverages Multimodal LLMs to improve surgical context comprehension via Self-Contained Inquiry. SCAN operates autonomously, generating two types of memory for context augmentation: Direct Memory (DM), which provides multiple candidates (or hints) to the final answer, and Indirect Memory (IM), which consists of self-contained question-hint pairs to capture broader scene context. DM directly assists in answering the question, while IM enhances understanding of the surgical scene beyond the immediate query. Reasoning over these object-aware memories enables the model to accurately interpret images and respond to questions. Extensive experiments on three publicly available Surgical VQA datasets demonstrate that SCAN achieves state-of-the-art performance, offering improved accuracy and robustness across various surgical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10934v1">Analyzing Pokémon and Mario Streamers' Twitch Chat with LLM-based User Embeddings</a></div>
    <div class="paper-meta">
      📅 2024-11-17
      | 💬 NLP4DH 2024
    </div>
    <details class="paper-abstract">
      We present a novel digital humanities method for representing our Twitch chatters as user embeddings created by a large language model (LLM). We cluster these embeddings automatically using affinity propagation and further narrow this clustering down through manual analysis. We analyze the chat of one stream by each Twitch streamer: SmallAnt, DougDoug and PointCrow. Our findings suggest that each streamer has their own type of chatters, however two categories emerge for all of the streamers: supportive viewers and emoji and reaction senders. Repetitive message spammers is a shared chatter category for two of the streamers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11798v2">PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 11 pages, submitted to SC24 conference
    </div>
    <details class="paper-abstract">
      Inference of Large Language Models (LLMs) across computer clusters has become a focal point of research in recent times, with many acceleration techniques taking inspiration from CPU speculative execution. These techniques reduce bottlenecks associated with memory bandwidth, but also increase end-to-end latency per inference run, requiring high speculation acceptance rates to improve performance. Combined with a variable rate of acceptance across tasks, speculative inference techniques can result in reduced performance. Additionally, pipeline-parallel designs require many user requests to maintain maximum utilization. As a remedy, we propose PipeInfer, a pipelined speculative acceleration technique to reduce inter-token latency and improve system utilization for single-request scenarios while also improving tolerance to low speculation acceptance rates and low-bandwidth interconnects. PipeInfer exhibits up to a 2.15$\times$ improvement in generation speed over standard speculative inference. PipeInfer achieves its improvement through Continuous Asynchronous Speculation and Early Inference Cancellation, the former improving latency and generation speed by running single-token inference simultaneously with several speculative runs, while the latter improves speed and latency by skipping the computation of invalidated runs, even in the middle of inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10869v1">Large Language Models (LLMs) as Traffic Control Systems at Urban Intersections: A New Paradigm</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 The data and code that support the findings of this study are openly available in Zenodo at https://doi.org/10.5281/zenodo.14171745, reference number 14171745
    </div>
    <details class="paper-abstract">
      This study introduces a novel approach for traffic control systems by using Large Language Models (LLMs) as traffic controllers. The study utilizes their logical reasoning, scene understanding, and decision-making capabilities to optimize throughput and provide feedback based on traffic conditions in real-time. LLMs centralize traditionally disconnected traffic control processes and can integrate traffic data from diverse sources to provide context-aware decisions. LLMs can also deliver tailored outputs using various means such as wireless signals and visuals to drivers, infrastructures, and autonomous vehicles. To evaluate LLMs ability as traffic controllers, this study proposed a four-stage methodology. The methodology includes data creation and environment initialization, prompt engineering, conflict identification, and fine-tuning. We simulated multi-lane four-leg intersection scenarios and generates detailed datasets to enable conflict detection using LLMs and Python simulation as a ground truth. We used chain-of-thought prompts to lead LLMs in understanding the context, detecting conflicts, resolving them using traffic rules, and delivering context-sensitive traffic management solutions. We evaluated the prformance GPT-mini, Gemini, and Llama as traffic controllers. Results showed that the fine-tuned GPT-mini achieved 83% accuracy and an F1-score of 0.84. GPT-mini model exhibited a promising performance in generating actionable traffic management insights, with high ROUGE-L scores across conflict identification of 0.95, decision-making of 0.91, priority assignment of 0.94, and waiting time optimization of 0.92. We demonstrated that LLMs can offer precise recommendations to drivers in real-time including yielding, slowing, or stopping based on vehicle dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15302v5">How (un)ethical are instruction-centric responses of LLMs? Unveiling the vulnerabilities of safety guardrails to harmful queries</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Accepted at AAAI Conference on Web and Social Media (ICWSM) 2025. [Dataset](https://huggingface.co/datasets/SoftMINER-Group/TechHazardQA)
    </div>
    <details class="paper-abstract">
      In this study, we tackle a growing concern around the safety and ethical use of large language models (LLMs). Despite their potential, these models can be tricked into producing harmful or unethical content through various sophisticated methods, including 'jailbreaking' techniques and targeted manipulation. Our work zeroes in on a specific issue: to what extent LLMs can be led astray by asking them to generate responses that are instruction-centric such as a pseudocode, a program or a software snippet as opposed to vanilla text. To investigate this question, we introduce TechHazardQA, a dataset containing complex queries which should be answered in both text and instruction-centric formats (e.g., pseudocodes), aimed at identifying triggers for unethical responses. We query a series of LLMs -- Llama-2-13b, Llama-2-7b, Mistral-V2 and Mistral 8X7B -- and ask them to generate both text and instruction-centric responses. For evaluation we report the harmfulness score metric as well as judgements from GPT-4 and humans. Overall, we observe that asking LLMs to produce instruction-centric responses enhances the unethical response generation by ~2-38% across the models. As an additional objective, we investigate the impact of model editing using the ROME technique, which further increases the propensity for generating undesirable content. In particular, asking edited LLMs to generate instruction-centric responses further increases the unethical response generation by ~3-16% across the different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05049v2">ProverbEval: Exploring LLM Evaluation Challenges for Low-resource Language Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      With the rapid development of evaluation datasets to assess LLMs understanding across a wide range of subjects and domains, identifying a suitable language understanding benchmark has become increasingly challenging. In this work, we explore LLM evaluation challenges for low-resource language understanding and introduce ProverbEval, LLM evaluation benchmark for low-resource languages based on proverbs to focus on low-resource language understanding in culture-specific scenarios. We benchmark various LLMs and explore factors that create variability in the benchmarking process. We observed performance variances of up to 50%, depending on the order in which answer choices were presented in multiple-choice tasks. Native language proverb descriptions significantly improve tasks such as proverb generation, contributing to improved outcomes. Additionally, monolingual evaluations consistently outperformed their cross-lingual counterparts. We argue special attention must be given to the order of choices, choice of prompt language, task variability, and generation tasks when creating LLM evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07021v2">Invar-RAG: Invariant LLM-aligned Retrieval for Better Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has shown impressive capability in providing reliable answer predictions and addressing hallucination problems. A typical RAG implementation uses powerful retrieval models to extract external information and large language models (LLMs) to generate answers. In contrast, recent LLM-based retrieval has gained attention for its substantial improvements in information retrieval (IR) due to the LLMs' semantic understanding capability. However, directly applying LLM to RAG systems presents challenges. This may cause feature locality problems as massive parametric knowledge can hinder effective usage of global information across the corpus; for example, an LLM-based retriever often inputs document summaries instead of full documents. Moreover, various pre-trained tasks in LLMs introduce variance, further weakening performance as a retriever. To address these issues, we propose a novel two-stage fine-tuning architecture called Invar-RAG. In the retrieval stage, an LLM-based retriever is constructed by integrating LoRA-based representation learning to tackle feature locality issues. To enhance retrieval performance, we develop two patterns (invariant and variant patterns) and an invariance loss to reduce LLM variance. In the generation stage, a refined fine-tuning method is employed to improve LLM accuracy in generating answers based on retrieved information. Experimental results show that Invar-RAG significantly outperforms existing baselines across three open-domain question answering (ODQA) datasets. Code is available in the Supplementary Material for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14459v1">Unveiling User Preferences: A Knowledge Graph and LLM-Driven Approach for Conversational Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      Conversational Recommender Systems (CRSs) aim to provide personalized recommendations through dynamically capturing user preferences in interactive conversations. Conventional CRSs often extract user preferences as hidden representations, which are criticized for their lack of interpretability. This diminishes the transparency and trustworthiness of the recommendation process. Recent works have explored combining the impressive capabilities of Large Language Models (LLMs) with the domain-specific knowledge of Knowledge Graphs (KGs) to generate human-understandable recommendation explanations. Despite these efforts, the integration of LLMs and KGs for CRSs remains challenging due to the modality gap between unstructured dialogues and structured KGs. Moreover, LLMs pre-trained on large-scale corpora may not be well-suited for analyzing user preferences, which require domain-specific knowledge. In this paper, we propose COMPASS, a plug-and-play framework that synergizes LLMs and KGs to unveil user preferences, enhancing the performance and explainability of existing CRSs. To address integration challenges, COMPASS employs a two-stage training approach: first, it bridges the gap between the structured KG and natural language through an innovative graph entity captioning pre-training mechanism. This enables the LLM to transform KG entities into concise natural language descriptions, allowing them to comprehend domain-specific knowledge. Following, COMPASS optimizes user preference modeling via knowledge-aware instruction fine-tuning, where the LLM learns to reason and summarize user preferences from both dialogue histories and KG-augmented context. This enables COMPASS to perform knowledge-aware reasoning and generate comprehensive and interpretable user preferences that can seamlessly integrate with existing CRS models for improving recommendation performance and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10761v1">Can Generic LLMs Help Analyze Child-adult Interactions Involving Children with Autism in Clinical Observation?</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 GenAI for Health Workshop, NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential in understanding human communication and interaction. However, their performance in the domain of child-inclusive interactions, including in clinical settings, remains less explored. In this work, we evaluate generic LLMs' ability to analyze child-adult dyadic interactions in a clinically relevant context involving children with ASD. Specifically, we explore LLMs in performing four tasks: classifying child-adult utterances, predicting engaged activities, recognizing language skills and understanding traits that are clinically relevant. Our evaluation shows that generic LLMs are highly capable of analyzing long and complex conversations in clinical observation sessions, often surpassing the performance of non-expert human evaluators. The results show their potential to segment interactions of interest, assist in language skills evaluation, identify engaged activities, and offer clinical-relevant context for assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12760v1">VayuBuddy: an LLM-Powered Chatbot to Democratize Air Quality Insights</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      Nearly 6.7 million lives are lost due to air pollution every year. While policymakers are working on the mitigation strategies, public awareness can help reduce the exposure to air pollution. Air pollution data from government-installed sensors is often publicly available in raw format, but there is a non-trivial barrier for various stakeholders in deriving meaningful insights from that data. In this work, we present VayuBuddy, a Large Language Model (LLM)-powered chatbot system to reduce the barrier between the stakeholders and air quality sensor data. VayuBuddy receives the questions in natural language, analyses the structured sensory data with a LLM-generated Python code and provides answers in natural language. We use the data from Indian government air quality sensors. We benchmark the capabilities of 7 LLMs on 45 diverse question-answer pairs prepared by us. Additionally, VayuBuddy can also generate visual analysis such as line-plots, map plot, bar charts and many others from the sensory data as we demonstrate in this work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16527v3">Insights and Current Gaps in Open-Source LLM Vulnerability Scanners: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      This report presents a comparative analysis of open-source vulnerability scanners for conversational large language models (LLMs). As LLMs become integral to various applications, they also present potential attack surfaces, exposed to security risks such as information leakage and jailbreak attacks. Our study evaluates prominent scanners - Garak, Giskard, PyRIT, and CyberSecEval - that adapt red-teaming practices to expose these vulnerabilities. We detail the distinctive features and practical use of these scanners, outline unifying principles of their design and perform quantitative evaluations to compare them. These evaluations uncover significant reliability issues in detecting successful attacks, highlighting a fundamental gap for future development. Additionally, we contribute a preliminary labelled dataset, which serves as an initial step to bridge this gap. Based on the above, we provide strategic recommendations to assist organizations choose the most suitable scanner for their red-teaming needs, accounting for customizability, test suite comprehensiveness, and industry-specific use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10696v1">HELENE: Hessian Layer-wise Clipping and Gradient Annealing for Accelerating Fine-tuning LLM with Zeroth-order Optimization</a></div>
    <div class="paper-meta">
      📅 2024-11-16
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) poses significant memory challenges, as the back-propagation process demands extensive resources, especially with growing model sizes. Recent work, MeZO, addresses this issue using a zeroth-order (ZO) optimization method, which reduces memory consumption by matching the usage to the inference phase. However, MeZO experiences slow convergence due to varying curvatures across model parameters. To overcome this limitation, we introduce HELENE, a novel scalable and memory-efficient optimizer that integrates annealed A-GNB gradients with a diagonal Hessian estimation and layer-wise clipping, serving as a second-order pre-conditioner. This combination allows for faster and more stable convergence. Our theoretical analysis demonstrates that HELENE improves convergence rates, particularly for models with heterogeneous layer dimensions, by reducing the dependency on the total parameter space dimension. Instead, the method scales with the largest layer dimension, making it highly suitable for modern LLM architectures. Experimental results on RoBERTa-large and OPT-1.3B across multiple tasks show that HELENE achieves up to a 20x speedup compared to MeZO, with average accuracy improvements of 1.5%. Furthermore, HELENE remains compatible with both full parameter tuning and parameter-efficient fine-tuning (PEFT), outperforming several state-of-the-art optimizers. The codes will be released after reviewing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10683v1">I'm Spartacus, No, I'm Spartacus: Measuring and Understanding LLM Identity Confusion</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 16 pages, 8 figure, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in diverse tasks such as text generation, data analysis, and software development, making them indispensable across domains like education, business, and creative industries. However, the rapid proliferation of LLMs (with over 560 companies developing or deploying them as of 2024) has raised concerns about their originality and trustworthiness. A notable issue, termed identity confusion, has emerged, where LLMs misrepresent their origins or identities. This study systematically examines identity confusion through three research questions: (1) How prevalent is identity confusion among LLMs? (2) Does it arise from model reuse, plagiarism, or hallucination? (3) What are the security and trust-related impacts of identity confusion? To address these, we developed an automated tool combining documentation analysis, self-identity recognition testing, and output similarity comparisons--established methods for LLM fingerprinting--and conducted a structured survey via Credamo to assess its impact on user trust. Our analysis of 27 LLMs revealed that 25.93% exhibit identity confusion. Output similarity analysis confirmed that these issues stem from hallucinations rather than replication or reuse. Survey results further highlighted that identity confusion significantly erodes trust, particularly in critical tasks like education and professional use, with declines exceeding those caused by logical errors or inconsistencies. Users attributed these failures to design flaws, incorrect training data, and perceived plagiarism, underscoring the systemic risks posed by identity confusion to LLM reliability and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10681v1">Structured Dialogue System for Mental Health: An LLM Chatbot Leveraging the PM+ Guidelines</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Accepted to the 16th International Conference on Social Robotic (ICSR 2024)
    </div>
    <details class="paper-abstract">
      The Structured Dialogue System, referred to as SuDoSys, is an innovative Large Language Model (LLM)-based chatbot designed to provide psychological counseling. SuDoSys leverages the World Health Organization (WHO)'s Problem Management Plus (PM+) guidelines to deliver stage-aware multi-turn dialogues. Existing methods for employing an LLM in multi-turn psychological counseling typically involve direct fine-tuning using generated dialogues, often neglecting the dynamic stage shifts of counseling sessions. Unlike previous approaches, SuDoSys considers the different stages of counseling and stores essential information throughout the counseling process, ensuring coherent and directed conversations. The system employs an LLM, a stage-aware instruction generator, a response unpacker, a topic database, and a stage controller to maintain dialogue flow. In addition, we propose a novel technique that simulates counseling clients to interact with the evaluated system and evaluate its performance automatically. When assessed using both objective and subjective evaluations, SuDoSys demonstrates its effectiveness in generating logically coherent responses. The system's code and program scripts for evaluation are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01768v2">Stereotype Detection in LLMs: A Multiclass, Explainable, and Benchmark-Driven Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-16
      | 💬 Under review as a conference paper at ARR October 2024
    </div>
    <details class="paper-abstract">
      Stereotype detection is a challenging and subjective task, as certain statements, such as "Black people like to play basketball," may not appear overtly toxic but still reinforce racial stereotypes. With the increasing prevalence of large language models (LLMs) in human-facing artificial intelligence (AI) applications, detecting these types of biases is essential. However, LLMs risk perpetuating and amplifying stereotypical outputs derived from their training data. A reliable stereotype detector is crucial for benchmarking bias, monitoring model input and output, filtering training data, and ensuring fairer model behavior in downstream applications. This paper introduces the Multi-Grain Stereotype (MGS) dataset, consisting of 51,867 instances across gender, race, profession, religion, and other stereotypes, curated from multiple existing datasets. We evaluate various machine learning approaches to establish baselines and fine-tune language models of different architectures and sizes, presenting a suite of stereotype multiclass classifiers trained on the MGS dataset. Given the subjectivity of stereotypes, explainability is essential to align model learning with human understanding of stereotypes. We employ explainable AI (XAI) tools, including SHAP, LIME, and BertViz, to assess whether the model's learned patterns align with human intuitions about stereotypes.Additionally, we develop stereotype elicitation prompts and benchmark the presence of stereotypes in text generation tasks using popular LLMs, employing the best-performing stereotype classifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10599v1">Generating Energy-efficient code with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      The increasing electricity demands of personal computers, communication networks, and data centers contribute to higher atmospheric greenhouse gas emissions, which in turn lead to global warming and climate change. Therefore the energy consumption of code must be minimized. Code can be generated by large language models. We look at the influence of prompt modification on the energy consumption of the code generated. We use three different Python code problems of varying difficulty levels. Prompt modification is done by adding the sentence ``Give me an energy-optimized solution for this problem'' or by using two Python coding best practices. The large language models used are CodeLlama-70b, CodeLlama-70b-Instruct, CodeLlama-70b-Python, DeepSeek-Coder-33b-base, and DeepSeek-Coder-33b-instruct. We find a decrease in energy consumption for a specific combination of prompt optimization, LLM, and Python code problem. However, no single optimization prompt consistently decreases energy consumption for the same LLM across the different Python code problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12591v1">Thinking Before Looking: Improving Multimodal LLM Reasoning via Mitigating Visual Hallucination</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have advanced the integration of visual and linguistic modalities, establishing themselves as the dominant paradigm for visual-language tasks. Current approaches like chain of thought (CoT) reasoning have augmented the cognitive capabilities of large language models (LLMs), yet their adaptation to MLLMs is hindered by heightened risks of hallucination in cross-modality comprehension. In this paper, we find that the thinking while looking paradigm in current multimodal CoT approaches--where reasoning chains are generated alongside visual input--fails to mitigate hallucinations caused by misleading images. To address these limitations, we propose the Visual Inference Chain (VIC) framework, a novel approach that constructs reasoning chains using textual context alone before introducing visual input, effectively reducing cross-modal biases and enhancing multimodal reasoning accuracy. Comprehensive evaluations demonstrate that VIC significantly improves zero-shot performance across various vision-related tasks, mitigating hallucinations while refining the reasoning capabilities of MLLMs. Our code repository can be found at https://github.com/Terry-Xu-666/visual_inference_chain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10565v1">Comparing Robustness Against Adversarial Attacks in Code Generation: LLM-Generated vs. Human-Written</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Thanks to the widespread adoption of Large Language Models (LLMs) in software engineering research, the long-standing dream of automated code generation has become a reality on a large scale. Nowadays, LLMs such as GitHub Copilot and ChatGPT are extensively used in code generation for enterprise and open-source software development and maintenance. Despite their unprecedented successes in code generation, research indicates that codes generated by LLMs exhibit vulnerabilities and security issues. Several studies have been conducted to evaluate code generated by LLMs, considering various aspects such as security, vulnerability, code smells, and robustness. While some studies have compared the performance of LLMs with that of humans in various software engineering tasks, there's a notable gap in research: no studies have directly compared human-written and LLM-generated code for their robustness analysis. To fill this void, this paper introduces an empirical study to evaluate the adversarial robustness of Pre-trained Models of Code (PTMCs) fine-tuned on code written by humans and generated by LLMs against adversarial attacks for software clone detection. These attacks could potentially undermine software security and reliability. We consider two datasets, two state-of-the-art PTMCs, two robustness evaluation criteria, and three metrics to use in our experiments. Regarding effectiveness criteria, PTMCs fine-tuned on human-written code always demonstrate more robustness than those fine-tuned on LLMs-generated code. On the other hand, in terms of adversarial code quality, in 75% experimental combinations, PTMCs fine-tuned on the human-written code exhibit more robustness than the PTMCs fine-tuned on the LLMs-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10541v1">Does Prompt Formatting Have Any Impact on LLM Performance?</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 Submitted to NAACL 2025
    </div>
    <details class="paper-abstract">
      In the realm of Large Language Models (LLMs), prompt optimization is crucial for model performance. Although previous research has explored aspects like rephrasing prompt contexts, using various prompting techniques (like in-context learning and chain-of-thought), and ordering few-shot examples, our understanding of LLM sensitivity to prompt templates remains limited. Therefore, this paper examines the impact of different prompt templates on LLM performance. We formatted the same contexts into various human-readable templates, including plain text, Markdown, JSON, and YAML, and evaluated their impact across tasks like natural language reasoning, code generation, and translation using OpenAI's GPT models. Experiments show that GPT-3.5-turbo's performance varies by up to 40\% in a code translation task depending on the prompt template, while larger models like GPT-4 are more robust to these variations. Our analysis highlights the need to reconsider the use of fixed prompt templates, as different formats can significantly affect model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05818v2">Open LLMs are Necessary for Current Private Adaptations and Outperform their Closed Alternatives</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 Accepted at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      While open Large Language Models (LLMs) have made significant progress, they still fall short of matching the performance of their closed, proprietary counterparts, making the latter attractive even for the use on highly private data. Recently, various new methods have been proposed to adapt closed LLMs to private data without leaking private information to third parties and/or the LLM provider. In this work, we analyze the privacy protection and performance of the four most recent methods for private adaptation of closed LLMs. By examining their threat models and thoroughly comparing their performance under different privacy levels according to differential privacy (DP), various LLM architectures, and multiple datasets for classification and generation tasks, we find that: (1) all the methods leak query data, i.e., the (potentially sensitive) user data that is queried at inference time, to the LLM provider, (2) three out of four methods also leak large fractions of private training data to the LLM provider while the method that protects private data requires a local open LLM, (3) all the methods exhibit lower performance compared to three private gradient-based adaptation methods for local open LLMs, and (4) the private adaptation methods for closed LLMs incur higher monetary training and query costs than running the alternative methods on local open LLMs. This yields the conclusion that, to achieve truly privacy-preserving LLM adaptations that yield high performance and more privacy at lower costs, taking into account current methods and models, one should use open LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17710v3">Optimization-based Prompt Injection Attack to LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 To appear in the Proceedings of The ACM Conference on Computer and Communications Security (CCS), 2024
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10213v1">An Empirical Study on LLM-based Agents for Automated Bug Fixing</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and LLM-based Agents have been applied to fix bugs automatically, demonstrating the capability in addressing software defects by engaging in development environment interaction, iterative validation and code modification. However, systematic analysis of these agent and non-agent systems remain limited, particularly regarding performance variations among top-performing ones. In this paper, we examine seven proprietary and open-source systems on the SWE-bench Lite benchmark for automated bug fixing. We first assess each system's overall performance, noting instances solvable by all or none of these sytems, and explore why some instances are uniquely solved by specific system types. We also compare fault localization accuracy at file and line levels and evaluate bug reproduction capabilities, identifying instances solvable only through dynamic reproduction. Through analysis, we concluded that further optimization is needed in both the LLM itself and the design of Agentic flow to improve the effectiveness of the Agent in bug fixing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10184v1">Agentic LLMs in the Supply Chain: Towards Autonomous Multi-Agent Consensus-Seeking</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      This paper explores how Large Language Models (LLMs) can automate consensus-seeking in supply chain management (SCM), where frequent decisions on problems such as inventory levels and delivery times require coordination among companies. Traditional SCM relies on human consensus in decision-making to avoid emergent problems like the bullwhip effect. Some routine consensus processes, especially those that are time-intensive and costly, can be automated. Existing solutions for automated coordination have faced challenges due to high entry barriers locking out SMEs, limited capabilities, and limited adaptability in complex scenarios. However, recent advances in Generative AI, particularly LLMs, show promise in overcoming these barriers. LLMs, trained on vast datasets can negotiate, reason, and plan, facilitating near-human-level consensus at scale with minimal entry barriers. In this work, we identify key limitations in existing approaches and propose autonomous LLM agents to address these gaps. We introduce a series of novel, supply chain-specific consensus-seeking frameworks tailored for LLM agents and validate the effectiveness of our approach through a case study in inventory management. To accelerate progress within the SCM community, we open-source our code, providing a foundation for further advancements in LLM-powered autonomous supply chain solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10163v1">Compound-QA: A Benchmark for Evaluating LLMs on Compound Questions</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance across various tasks, prompting researchers to develop diverse evaluation benchmarks. However, existing benchmarks typically measure the ability of LLMs to respond to individual questions, neglecting the complex interactions in real-world applications. In this paper, we introduce Compound Question Synthesis (CQ-Syn) to create the Compound-QA benchmark, focusing on compound questions with multiple sub-questions. This benchmark is derived from existing QA datasets, annotated with proprietary LLMs and verified by humans for accuracy. It encompasses five categories: Factual-Statement, Cause-and-Effect, Hypothetical-Analysis, Comparison-and-Selection, and Evaluation-and-Suggestion. It evaluates the LLM capability in terms of three dimensions including understanding, reasoning, and knowledge. Our assessment of eight open-source LLMs using Compound-QA reveals distinct patterns in their responses to compound questions, which are significantly poorer than those to non-compound questions. Additionally, we investigate various methods to enhance LLMs performance on compound questions. The results indicate that these approaches significantly improve the models' comprehension and reasoning abilities on compound questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09510v2">Communication Compression for Tensor Parallel LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have pushed the frontier of artificial intelligence but are comprised of hundreds of billions of parameters and operations. For faster inference latency, LLMs are deployed on multiple hardware accelerators through various Model Parallelism strategies. Our paper looks into the details on one such strategy - Tensor Parallel - and proposes to reduce latency by compressing inter-accelerator communication. We leverage fine grained quantization techniques to compress selected activations by 3.5 - 4.5x. Our proposed method leads up to 2x reduction of time-to-first-token (TTFT) with negligible model performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09978v1">HistoLens: An LLM-Powered Framework for Multi-Layered Analysis of Historical Texts -- A Case Application of Yantie Lun</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      This paper proposes HistoLens, a multi-layered analysis framework for historical texts based on Large Language Models (LLMs). Using the important Western Han dynasty text "Yantie Lun" as a case study, we demonstrate the framework's potential applications in historical research and education. HistoLens integrates NLP technology (especially LLMs), including named entity recognition, knowledge graph construction, and geographic information visualization. The paper showcases how HistoLens explores Western Han culture in "Yantie Lun" through multi-dimensional, visual, and quantitative methods, focusing particularly on the influence of Confucian and Legalist thoughts on political, economic, military, and ethnic. We also demonstrate how HistoLens constructs a machine teaching scenario using LLMs for explainable analysis, based on a dataset of Confucian and Legalist ideas extracted with LLM assistance. This approach offers novel and diverse perspectives for studying historical texts like "Yantie Lun" and provides new auxiliary tools for history education. The framework aims to equip historians and learners with LLM-assisted tools to facilitate in-depth, multi-layered analysis of historical texts and foster innovation in historical education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02170v2">A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 Published in COLM2024. Code Repo: https://github.com/SALT-NLP/DyLAN
    </div>
    <details class="paper-abstract">
      Recent studies show that collaborating multiple large language model (LLM) powered agents is a promising way for task solving. However, current approaches are constrained by using a fixed number of agents and static communication structures. In this work, we propose automatically selecting a team of agents from candidates to collaborate in a dynamic communication structure toward different tasks and domains. Specifically, we build a framework named Dynamic LLM-Powered Agent Network ($\textbf{DyLAN}$) for LLM-powered agent collaboration, operating a two-stage paradigm: (1) Team Optimization and (2) Task Solving. During the first stage, we utilize an $\textit{agent selection}$ algorithm, based on an unsupervised metric called $\textit{Agent Importance Score}$, enabling the selection of best agents according to their contributions in a preliminary trial, oriented to the given task. Then, in the second stage, the selected agents collaborate dynamically according to the query. Empirically, we demonstrate that DyLAN outperforms strong baselines in code generation, decision-making, general reasoning, and arithmetic reasoning tasks with moderate computational cost. On specific subjects in MMLU, selecting a team of agents in the team optimization stage improves accuracy by up to 25.0% in DyLAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09916v1">LLMs are Imperfect, Then What? An Empirical Study on LLM Failures in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Software engineers are integrating AI assistants into their workflows to enhance productivity and reduce cognitive strain. However, experiences vary significantly, with some engineers finding large language models (LLMs), like ChatGPT, beneficial, while others consider them counterproductive. Researchers also found that ChatGPT's answers included incorrect information. Given the fact that LLMs are still imperfect, it is important to understand how to best incorporate LLMs into the workflow for software engineering (SE) task completion. Therefore, we conducted an observational study with 22 participants using ChatGPT as a coding assistant in a non-trivial SE task to understand the practices, challenges, and opportunities for using LLMs for SE tasks. We identified the cases where ChatGPT failed, their root causes, and the corresponding mitigation solutions used by users. These findings contribute to the overall understanding and strategies for human-AI interaction on SE tasks. Our study also highlights future research and tooling support directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09909v1">AMXFP4: Taming Activation Outliers with Asymmetric Microscaling Floating-Point for 4-bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Scaling Large Language Models (LLMs) with extended context lengths has increased the need for efficient low-bit quantization to manage their substantial computational demands. However, reducing precision to 4 bits frequently degrades performance due to activation outliers. To address this, we propose Asymmetric Microscaling 4-bit Floating-Point (AMXFP4) for efficient LLM inference. This novel data format leverages asymmetric shared scales to mitigate outliers while naturally capturing the asymmetry introduced by group-wise quantization. Unlike conventional 4-bit quantization methods that rely on data rotation and costly calibration, AMXFP4 uses asymmetric shared scales for direct 4-bit casting, achieving near-ideal quantization accuracy across various LLM tasks, including multi-turn conversations, long-context reasoning, and visual question answering. Our AMXFP4 format significantly outperforms MXFP4 and other leading quantization techniques, enabling robust, calibration-free 4-bit inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09410v2">LLM-assisted Explicit and Implicit Multi-interest Learning Framework for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Multi-interest modeling in current recommender systems (RS) is mainly based on user behavioral data, capturing user interest preferences from multiple dimensions. However, since behavioral data is implicit and often highly sparse, it is challenging to understand users' complex and diverse interests. Recent studies have shown that the rich semantic information in the text can effectively supplement the deficiencies of behavioral data. Despite this, it is still difficult for small models to directly extract semantic features associated with users' deep interests. That is, how to effectively align semantics with behavioral information to form a more comprehensive and accurate understanding of user interests has become a critical research problem. To address this, we propose an LLM-assisted explicit and implicit multi-interest learning framework (named EIMF) to model user interests on two levels: behavior and semantics. The framework consists of two parts: Implicit Behavioral Interest Module (IBIM) and Explicit Semantic Interest Module (ESIM). The traditional multi-interest RS model in IBIM can learn users' implicit behavioral interests from interactions with items. In ESIM, we first adopt a clustering algorithm to select typical samples and design a prompting strategy on LLM to obtain explicit semantic interests. Furthermore, in the training phase, the semantic interests of typical samples can enhance the representation learning of behavioral interests based on the multi-task learning on semantic prediction and modality alignment. Therefore, in the inference stage, accurate recommendations can be achieved with only the user's behavioral data. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed EIMF framework, which effectively and efficiently combines small models with LLM to improve the accuracy of multi-interest modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18027v2">Automated Clinical Data Extraction with Knowledge Conditioned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-15
      | 💬 COLING25 Industry Track
    </div>
    <details class="paper-abstract">
      The extraction of lung lesion information from clinical and medical imaging reports is crucial for research on and clinical care of lung-related diseases. Large language models (LLMs) can be effective at interpreting unstructured text in reports, but they often hallucinate due to a lack of domain-specific knowledge, leading to reduced accuracy and posing challenges for use in clinical settings. To address this, we propose a novel framework that aligns generated internal knowledge with external knowledge through in-context learning (ICL). Our framework employs a retriever to identify relevant units of internal or external knowledge and a grader to evaluate the truthfulness and helpfulness of the retrieved internal-knowledge rules, to align and update the knowledge bases. Experiments with expert-curated test datasets demonstrate that this ICL approach can increase the F1 score for key fields (lesion size, margin and solidity) by an average of 12.9% over existing ICL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09873v1">LLM-Powered AI Tutors with Personas for d/Deaf and Hard-of-Hearing Online Learners</a></div>
    <div class="paper-meta">
      📅 2024-11-15
    </div>
    <details class="paper-abstract">
      Intelligent tutoring systems (ITS) using artificial intelligence (AI) technology have shown promise in supporting learners with diverse abilities; however, they often fail to meet the specific communication needs and cultural nuances needed by d/Deaf and Hard-of-Hearing (DHH) learners. As large language models (LLMs) provide new opportunities to incorporate personas to AI-based tutors and support dynamic interactive dialogue, this paper explores how DHH learners perceive LLM-powered ITS with different personas and identified design suggestions for improving the interaction. We developed an interface that allows DHH learners to interact with ChatGPT and three LLM-powered AI tutors with different experiences in DHH education while the learners watch an educational video. A user study with 16 DHH participants showed that they perceived conversations with the AI tutors who had DHH education experiences to be more human-like and trustworthy due to the tutors' cultural knowledge of DHH communities. Participants also suggested providing more transparency regarding the tutors' background information to clarify each AI tutor's position within the DHH community. We discuss design implications for more inclusive LLM-based systems, such as supports for the multimodality of sign language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08123v2">Exploring the Role of LLMs for Supporting Older Adults: Opportunities and Concerns</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 This short paper was accepted at CHI 2024 Workshop on HCI and Aging: New Directions, New Principles
    </div>
    <details class="paper-abstract">
      We explore some of the existing research in HCI around technology for older adults and examine the role of LLMs in enhancing it. We also discuss the digital divide and emphasize the need for inclusive technology design. At the same time, we also surface concerns regarding privacy, security, and the accuracy of information provided by LLMs, alongside the importance of user-centered design to make technology accessible and effective for the elderly. We show the transformative possibilities of LLM-supported interactions at the intersection of aging, technology, and human-computer interaction, advocating for further research and development in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09689v1">LLM Hallucination Reasoning with Zero-shot Knowledge Test</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      LLM hallucination, where LLMs occasionally generate unfaithful text, poses significant challenges for their practical applications. Most existing detection methods rely on external knowledge, LLM fine-tuning, or hallucination-labeled datasets, and they do not distinguish between different types of hallucinations, which are crucial for improving detection performance. We introduce a new task, Hallucination Reasoning, which classifies LLM-generated text into one of three categories: aligned, misaligned, and fabricated. Our novel zero-shot method assesses whether LLM has enough knowledge about a given prompt and text. Our experiments conducted on new datasets demonstrate the effectiveness of our method in hallucination reasoning and underscore its importance for enhancing detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04783v2">AutoDefense: Multi-Agent LLM Defense against Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Despite extensive pre-training in moral alignment to prevent generating harmful information, large language models (LLMs) remain vulnerable to jailbreak attacks. In this paper, we propose AutoDefense, a multi-agent defense framework that filters harmful responses from LLMs. With the response-filtering mechanism, our framework is robust against different jailbreak attack prompts, and can be used to defend different victim models. AutoDefense assigns different roles to LLM agents and employs them to complete the defense task collaboratively. The division in tasks enhances the overall instruction-following of LLMs and enables the integration of other defense components as tools. With AutoDefense, small open-source LMs can serve as agents and defend larger models against jailbreak attacks. Our experiments show that AutoDefense can effectively defense against different jailbreak attacks, while maintaining the performance at normal user request. For example, we reduce the attack success rate on GPT-3.5 from 55.74% to 7.95% using LLaMA-2-13b with a 3-agent system. Our code and data are publicly available at https://github.com/XHMY/AutoDefense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09590v1">Adopting RAG for LLM-Aided Future Vehicle Design</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 Conference paper accepted in IEEE FLLM 2024
    </div>
    <details class="paper-abstract">
      In this paper, we explore the integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) to enhance automated design and software development in the automotive industry. We present two case studies: a standardization compliance chatbot and a design copilot, both utilizing RAG to provide accurate, context-aware responses. We evaluate four LLMs-GPT-4o, LLAMA3, Mistral, and Mixtral -- comparing their answering accuracy and execution time. Our results demonstrate that while GPT-4 offers superior performance, LLAMA3 and Mistral also show promising capabilities for local deployment, addressing data privacy concerns in automotive applications. This study highlights the potential of RAG-augmented LLMs in improving design workflows and compliance in automotive engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09439v1">Spider: Any-to-Many Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) have emerged as an extension of Large Language Models (LLMs), enabling the integration of various modalities. However, Any-to-Any MLLMs are limited to generating pairwise modalities 'Text + X' within a single response, such as Text + {Image or Audio or Video}. To address this limitation, we introduce Spider, a novel efficient Any-to-Many Modalities Generation (AMMG) framework, which can generate an arbitrary combination of modalities 'Text + Xs', such as Text + {Image and Audio and Video}. To achieve efficient AMMG, our Spider integrates three core components: a Base Model for basic X-to-X (i.e., Any-to-Any) modality processing, a novel Efficient Decoders-Controller for controlling multimodal Decoders to generate Xs (many-modal) contents, and an Any-to-Many Instruction Template designed for producing Xs signal prompts. To train Spider, we constructed a novel Text-formatted Many-Modal (TMM) dataset, which facilitates the learning of the X-to-Xs (i.e., Any-to-Many) capability necessary for AMMG. Ultimately, the well-trained Spider generates a pseudo X-to-Xs dataset, the first-ever X-to-Xs many-modal dataset, enhancing the potential for AMMG task in future research. Overall, this work not only pushes the boundary of multimodal interaction but also provides rich data support for advancing the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09523v1">Navigating the Risks: A Survey of Security, Privacy, and Ethics Threats in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      With the continuous development of large language models (LLMs), transformer-based models have made groundbreaking advances in numerous natural language processing (NLP) tasks, leading to the emergence of a series of agents that use LLMs as their control hub. While LLMs have achieved success in various tasks, they face numerous security and privacy threats, which become even more severe in the agent scenarios. To enhance the reliability of LLM-based applications, a range of research has emerged to assess and mitigate these risks from different perspectives. To help researchers gain a comprehensive understanding of various risks, this survey collects and analyzes the different threats faced by these agents. To address the challenges posed by previous taxonomies in handling cross-module and cross-stage threats, we propose a novel taxonomy framework based on the sources and impacts. Additionally, we identify six key features of LLM-based agents, based on which we summarize the current research progress and analyze their limitations. Subsequently, we select four representative agents as case studies to analyze the risks they may face in practical use. Finally, based on the aforementioned analyses, we propose future research directions from the perspectives of data, methodology, and policy, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09492v1">MM-Eval: A Hierarchical Benchmark for Modern Mongolian Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in high-resource languages but face notable challenges in low-resource languages like Mongolian. This paper addresses these challenges by categorizing capabilities into language abilities (syntax and semantics) and cognitive abilities (knowledge and reasoning). To systematically evaluate these areas, we developed MM-Eval, a specialized dataset based on Modern Mongolian Language Textbook I and enriched with WebQSP and MGSM datasets. Preliminary experiments on models including Qwen2-7B-Instruct, GLM4-9b-chat, Llama3.1-8B-Instruct, GPT-4, and DeepseekV2.5 revealed that: 1) all models performed better on syntactic tasks than semantic tasks, highlighting a gap in deeper language understanding; and 2) knowledge tasks showed a moderate decline, suggesting that models can transfer general knowledge from high-resource to low-resource contexts. The release of MM-Eval, comprising 569 syntax, 677 semantics, 344 knowledge, and 250 reasoning tasks, offers valuable insights for advancing NLP and LLMs in low-resource languages like Mongolian. The dataset is available at https://github.com/joenahm/MM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06900v5">Can LLMs Recognize Toxicity? A Structured Investigation Framework and Toxicity Metric</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 8 page long
    </div>
    <details class="paper-abstract">
      In the pursuit of developing Large Language Models (LLMs) that adhere to societal standards, it is imperative to detect the toxicity in the generated text. The majority of existing toxicity metrics rely on encoder models trained on specific toxicity datasets, which are susceptible to out-of-distribution (OOD) problems and depend on the dataset's definition of toxicity. In this paper, we introduce a robust metric grounded on LLMs to flexibly measure toxicity according to the given definition. We first analyze the toxicity factors, followed by an examination of the intrinsic toxic attributes of LLMs to ascertain their suitability as evaluators. Finally, we evaluate the performance of our metric with detailed analysis. Our empirical results demonstrate outstanding performance in measuring toxicity within verified factors, improving on conventional metrics by 12 points in the F1 score. Our findings also indicate that upstream toxicity significantly influences downstream metrics, suggesting that LLMs are unsuitable for toxicity evaluations within unverified factors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07668v2">Towards Evaluation Guidelines for Empirical Studies involving LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      In the short period since the release of ChatGPT in November 2022, large language models (LLMs) have changed the software engineering research landscape. While there are numerous opportunities to use LLMs for supporting research or software engineering tasks, solid science needs rigorous empirical evaluations. However, so far, there are no specific guidelines for conducting and assessing studies involving LLMs in software engineering research. Our focus is on empirical studies that either use LLMs as part of the research process (e.g., for data annotation) or studies that evaluate existing or new tools that are based on LLMs. This paper contributes the first set of guidelines for such studies. Our goal is to start a discussion in the software engineering research community to reach a common understanding of what our community standards are for high-quality empirical studies involving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18406v2">IRCAN: Mitigating Knowledge Conflicts in LLM Generation via Identifying and Reweighting Context-Aware Neurons</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      It is widely acknowledged that large language models (LLMs) encode a vast reservoir of knowledge after being trained on mass data. Recent studies disclose knowledge conflicts in LLM generation, wherein outdated or incorrect parametric knowledge (i.e., encoded knowledge) contradicts new knowledge provided in the context. To mitigate such knowledge conflicts, we propose a novel framework, IRCAN (Identifying and Reweighting Context-Aware Neurons) to capitalize on neurons that are crucial in processing contextual cues. Specifically, IRCAN first identifies neurons that significantly contribute to context processing, utilizing a context-aware attribution score derived from integrated gradients. Subsequently, the identified context-aware neurons are strengthened via reweighting. In doing so, we steer LLMs to generate context-sensitive outputs with respect to the new knowledge provided in the context. Extensive experiments conducted across a variety of models and tasks demonstrate that IRCAN not only achieves remarkable improvements in handling knowledge conflicts but also offers a scalable, plug-and-play solution that can be integrated seamlessly with existing models. Our codes are released at https://github.com/danshi777/IRCAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09317v1">Pie: Pooling CPU Memory for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      The rapid growth of LLMs has revolutionized natural language processing and AI analysis, but their increasing size and memory demands present significant challenges. A common solution is to spill over to CPU memory; however, traditional GPU-CPU memory swapping often results in higher latency and lower throughput. This paper introduces Pie, an LLM inference framework that addresses these challenges with performance-transparent swapping and adaptive expansion. By leveraging predictable memory access patterns and the high bandwidth of modern hardware like the NVIDIA GH200 Grace Hopper Superchip, Pie enables concurrent data swapping without affecting foreground computation, expanding effective memory without added latency. Adaptive expansion dynamically adjusts CPU memory allocation based on real-time information, optimizing memory usage and performance under varying conditions. Pie maintains low computation latency, high throughput, and high elasticity. Our experimental evaluation demonstrates that Pie achieves optimal swapping policy during cache warmup and effectively balances increased memory capacity with negligible impact on computation. With its extended capacity, Pie outperforms vLLM by up to 1.9X in throughput and 2X in latency. Additionally, Pie can reduce GPU memory usage by up to 1.67X while maintaining the same performance. Compared to FlexGen, an offline profiling-based swapping solution, Pie achieves magnitudes lower latency and 9.4X higher throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09269v1">Harnessing multiple LLMs for Information Retrieval: A case study on Deep Learning methodologies in Biodiversity publications</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Deep Learning (DL) techniques are increasingly applied in scientific studies across various domains to address complex research questions. However, the methodological details of these DL models are often hidden in the unstructured text. As a result, critical information about how these models are designed, trained, and evaluated is challenging to access and comprehend. To address this issue, in this work, we use five different open-source Large Language Models (LLMs): Llama-3 70B, Llama-3.1 70B, Mixtral-8x22B-Instruct-v0.1, Mixtral 8x7B, and Gemma 2 9B in combination with Retrieval-Augmented Generation (RAG) approach to extract and process DL methodological details from scientific publications automatically. We built a voting classifier from the outputs of five LLMs to accurately report DL methodological information. We tested our approach using biodiversity publications, building upon our previous research. To validate our pipeline, we employed two datasets of DL-related biodiversity publications: a curated set of 100 publications from our prior work and a set of 364 publications from the Ecological Informatics journal. Our results demonstrate that the multi-LLM, RAG-assisted pipeline enhances the retrieval of DL methodological information, achieving an accuracy of 69.5% (417 out of 600 comparisons) based solely on textual content from publications. This performance was assessed against human annotators who had access to code, figures, tables, and other supplementary information. Although demonstrated in biodiversity, our methodology is not limited to this field; it can be applied across other scientific domains where detailed methodological reporting is essential for advancing knowledge and ensuring reproducibility. This study presents a scalable and reliable approach for automating information extraction, facilitating better reproducibility and knowledge transfer across studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06493v2">LProtector: An LLM-driven Vulnerability Detection System</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 5 pages, 4 figures. This is a preprint version of the article. The final version will be published in the proceedings of the IEEE conference
    </div>
    <details class="paper-abstract">
      This paper presents LProtector, an automated vulnerability detection system for C/C++ codebases driven by the large language model (LLM) GPT-4o and Retrieval-Augmented Generation (RAG). As software complexity grows, traditional methods face challenges in detecting vulnerabilities effectively. LProtector leverages GPT-4o's powerful code comprehension and generation capabilities to perform binary classification and identify vulnerabilities within target codebases. We conducted experiments on the Big-Vul dataset, showing that LProtector outperforms two state-of-the-art baselines in terms of F1 score, demonstrating the potential of integrating LLMs with vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09116v1">P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-14
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) showcase varied multilingual capabilities across tasks like translation, code generation, and reasoning. Previous assessments often limited their scope to fundamental natural language processing (NLP) or isolated capability-specific tasks. To alleviate this drawback, we aim to present a comprehensive multilingual multitask benchmark. First, we present a pipeline for selecting available and reasonable benchmarks from massive ones, addressing the oversight in previous work regarding the utility of these benchmarks, i.e., their ability to differentiate between models being evaluated. Leveraging this pipeline, we introduce P-MMEval, a large-scale benchmark covering effective fundamental and capability-specialized datasets. Furthermore, P-MMEval delivers consistent language coverage across various datasets and provides parallel samples. Finally, we conduct extensive experiments on representative multilingual model series to compare performances across models, analyze dataset effectiveness, examine prompt impacts on model performances, and explore the relationship between multilingual performances and factors such as tasks, model sizes, and languages. These insights offer valuable guidance for future research. The dataset is available at https://huggingface.co/datasets/Qwen/P-MMEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2403.15736v2">General LLMs as Instructors for Domain-Specific LLMs: A Sequential Fusion Method to Integrate Extraction and Editing</a></div>
    <div class="paper-meta">
      📅 2024-11-14
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      The substantial interest in updating Large Language Models (LLMs) without retraining from scratch is accompanied by several challenges. This is particularly true when updating LLMs with datasets that necessitate domain-expert reasoning across extensive texts, despite limited samples. We termed the scenario as the Few-Shot Domain-Expert Reasoning for Updating LLMs (FDoR-UL). Traditional methods such as Low-Rank Adaptation (LoRA) and Retrieval Augmented Generation (RAG) are inadequate for addressing this critical issue, particularly evident in our exploration of a specific medical dataset that epitomizes the distinct needs of FDoR-UL. To tackle this challenge, we introduce a Sequential Fusion method to integrate knowledge from complex contexts into LLMs. This method employs a two-stage framework: initially leveraging general LLMs to perform relation extraction for knowledge acquisition from complex texts, followed by updating domain-specific LLMs through Knowledge Editing (KE). Employing our method, domain-specific LLMs achieved a 71.7% accuracy (an average gain of 39.1%) in question-answering tasks. Furthermore, we expanded our evaluation to a novel economics-management dataset we developed, where our method achieved a 75.0% accuracy (an average gain of 45.0%). These findings underscore the effectiveness and flexibility of our approach in FDoR-UL across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08717v3">Fusing Dynamics Equation: A Social Opinions Prediction Algorithm with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 Submitted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      In the context where social media is increasingly becoming a significant platform for social movements and the formation of public opinion, accurately simulating and predicting the dynamics of user opinions is of great importance for understanding social phenomena, policy making, and guiding public opinion. However, existing simulation methods face challenges in capturing the complexity and dynamics of user behavior. Addressing this issue, this paper proposes an innovative simulation method for the dynamics of social media user opinions, the FDE-LLM algorithm, which incorporates opinion dynamics and epidemic model. This effectively constrains the actions and opinion evolution process of large language models (LLM), making them more aligned with the real cyber world. In particular, the FDE-LLM categorizes users into opinion leaders and followers. Opinion leaders are based on LLM role-playing and are constrained by the CA model, while opinion followers are integrated into a dynamic system that combines the CA model with the SIR model. This innovative design significantly improves the accuracy and efficiency of the simulation. Experiments were conducted on four real Weibo datasets and validated using the open-source model ChatGLM. The results show that, compared to traditional agent-based modeling (ABM) opinion dynamics algorithms and LLM-based opinion diffusion algorithms, our FDE-LLM algorithm demonstrates higher accuracy and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08294v1">Collaborative Participatory Research with LLM Agents in South Asia: An Empirically-Grounded Methodological Initiative and Agenda from Field Evidence in Sri Lanka</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The integration of artificial intelligence into development research methodologies presents unprecedented opportunities for addressing persistent challenges in participatory research, particularly in linguistically diverse regions like South Asia. Drawing from an empirical implementation in Sri Lanka's Sinhala-speaking communities, this paper presents an empirically grounded methodological framework designed to transform participatory development research, situated in the challenging multilingual context of Sri Lanka's flood-prone Nilwala River Basin. Moving beyond conventional translation and data collection tools, this framework deploys a multi-agent system architecture that redefines how data collection, analysis, and community engagement are conducted in linguistically and culturally diverse research settings. This structured agent-based approach enables participatory research that is both scalable and responsive, ensuring that community perspectives remain integral to research outcomes. Field experiences reveal the immense potential of LLM-based systems in addressing long-standing issues in development research across resource-limited regions, offering both quantitative efficiencies and qualitative improvements in inclusivity. At a broader methodological level, this research agenda advocates for AI-driven participatory research tools that maintain ethical considerations, cultural respect, and operational efficiency, highlighting strategic pathways for deploying AI systems that reinforce community agency and equitable knowledge generation, potentially informing broader research agendas across the Global South.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18164v2">Data-Prep-Kit: getting your data ready for LLM application development</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Data preparation is the first and a very important step towards any Large Language Model (LLM) development. This paper introduces an easy-to-use, extensible, and scale-flexible open-source data preparation toolkit called Data Prep Kit (DPK). DPK is architected and designed to enable users to scale their data preparation to their needs. With DPK they can prepare data on a local machine or effortlessly scale to run on a cluster with thousands of CPU Cores. DPK comes with a highly scalable, yet extensible set of modules that transform natural language and code data. If the user needs additional transforms, they can be easily developed using extensive DPK support for transform creation. These modules can be used independently or pipelined to perform a series of operations. In this paper, we describe DPK architecture and show its performance from a small scale to a very large number of CPUs. The modules from DPK have been used for the preparation of Granite Models [1] [2]. We believe DPK is a valuable contribution to the AI community to easily prepare data to enhance the performance of their LLM models or to fine-tune models with Retrieval-Augmented Generation (RAG).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08257v1">GPTree: Towards Explainable Decision-Making via LLM-powered Decision Trees</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Traditional decision tree algorithms are explainable but struggle with non-linear, high-dimensional data, limiting its applicability in complex decision-making. Neural networks excel at capturing complex patterns but sacrifice explainability in the process. In this work, we present GPTree, a novel framework combining explainability of decision trees with the advanced reasoning capabilities of LLMs. GPTree eliminates the need for feature engineering and prompt chaining, requiring only a task-specific prompt and leveraging a tree-based structure to dynamically split samples. We also introduce an expert-in-the-loop feedback mechanism to further enhance performance by enabling human intervention to refine and rebuild decision paths, emphasizing the harmony between human expertise and machine intelligence. Our decision tree achieved a 7.8% precision rate for identifying "unicorn" startups at the inception stage of a startup, surpassing gpt-4o with few-shot learning as well as the best human decision-makers (3.1% to 5.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09073v1">Code-mixed LLM: Improve Large Language Models' Capability to Handle Code-Mixing through Reinforcement Learning from AI Feedback</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 initial version: 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Code-mixing(CM) or code-switching(CSW) refers to the juxtaposition of linguistic units from two or more languages during the conversation or sometimes even a single utterance. Code-mixing introduces unique challenges in daily life, such as syntactic mismatches and semantic blending, that are rarely encountered in monolingual settings. Large language models (LLMs) have revolutionized the field of natural language processing (NLP) by offering unprecedented capabilities in understanding human languages. However, the effectiveness of current state-of-the-art multilingual LLMs has not yet been fully explored in the CM scenario. To fill this gap, we first benchmark the performance of multilingual LLMs on various code-mixing NLP tasks. Then we propose to improve the multilingual LLMs' ability to understand code-mixing through reinforcement learning from human feedback (RLHF) and code-mixed machine translation tasks. Given the high-cost and time-consuming preference labeling procedure, we improve this by utilizing LLMs as annotators to perform the reinforcement learning from AI feedback (RLAIF). The experiments show the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08979v1">CoCoP: Enhancing Text Classification with LLM through Code Completion Prompt</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Text classification is a fundamental task in natural language processing (NLP), and large language models (LLMs) have demonstrated their capability to perform this task across various domains. However, the performance of LLMs heavily depends on the quality of their input prompts. Recent studies have also shown that LLMs exhibit remarkable results in code-related tasks. To leverage the capabilities of LLMs in text classification, we propose the Code Completion Prompt (CoCoP) method, which transforms the text classification problem into a code completion task. CoCoP significantly improves text classification performance across diverse datasets by utilizing LLMs' code-completion capability. For instance, CoCoP enhances the accuracy of the SST2 dataset by more than 20%. Moreover, when CoCoP integrated with LLMs specifically designed for code-related tasks (code models), such as CodeLLaMA, this method demonstrates better or comparable performance to few-shot learning techniques while using only one-tenth of the model size. The source code of our proposed method will be available to the public upon the acceptance of the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08862v1">LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 Accepted at AAAI 2025
    </div>
    <details class="paper-abstract">
      We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08813v1">Rethinking CyberSecEval: An LLM-Aided Approach to Evaluation Critique</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 NeurIPS 2024, 2 pages
    </div>
    <details class="paper-abstract">
      A key development in the cybersecurity evaluations space is the work carried out by Meta, through their CyberSecEval approach. While this work is undoubtedly a useful contribution to a nascent field, there are notable features that limit its utility. Key drawbacks focus on the insecure code detection part of Meta's methodology. We explore these limitations, and use our exploration as a test case for LLM-assisted benchmark analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08794v1">Evaluating World Models with LLM for Decision Making</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      World model emerges as a key module in decision making, where MuZero and Dreamer achieve remarkable successes in complex tasks. Recent work leverages Large Language Models (LLMs) as general world simulators to simulate the dynamics of the world due to their generalizability. LLMs also serve as the world model for deliberative reasoning in Reasoning via Planning (RAP) and Tree of Thought (ToT). However, the world models are either evaluated as a general world simulator, or as a functional module of the agent, i.e., predicting the transitions to assist the planning. In this work, we propose a comprehensive evaluation of the world models with LLMs from the decision making perspective. Specifically, we leverage the 31 diverse environments from (Wang et al., 2023;2024) and curate the rule-based policy of each environment for the diverse evaluation. Then, we design three main tasks, i.e., policy verification, action proposal, and policy planning, where the world models can be used for decision making solely. Finally, we conduct the comprehensive evaluation of the advanced LLMs, i.e., GPT-4o and GPT-4o-mini, on the environments for the three main tasks under various settings. The key observations include: i) GPT-4o significantly outperforms GPT-4o-mini on the three main tasks, especially for the tasks which require the domain knowledge, ii) the performance of the world model with LLM will be decreased for long-term decision-making tasks, and iii) the combination of different functionalities of the world model will brings additional unstabilities of the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08696v1">Scholarly Wikidata: Population and Exploration of Conference Data in Wikidata using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 17 pages, accepted at EKAW-24
    </div>
    <details class="paper-abstract">
      Several initiatives have been undertaken to conceptually model the domain of scholarly data using ontologies and to create respective Knowledge Graphs. Yet, the full potential seems unleashed, as automated means for automatic population of said ontologies are lacking, and respective initiatives from the Semantic Web community are not necessarily connected: we propose to make scholarly data more sustainably accessible by leveraging Wikidata's infrastructure and automating its population in a sustainable manner through LLMs by tapping into unstructured sources like conference Web sites and proceedings texts as well as already existing structured conference datasets. While an initial analysis shows that Semantic Web conferences are only minimally represented in Wikidata, we argue that our methodology can help to populate, evolve and maintain scholarly data as a community within Wikidata. Our main contributions include (a) an analysis of ontologies for representing scholarly data to identify gaps and relevant entities/properties in Wikidata, (b) semi-automated extraction -- requiring (minimal) manual validation -- of conference metadata (e.g., acceptance rates, organizer roles, programme committee members, best paper awards, keynotes, and sponsors) from websites and proceedings texts using LLMs. Finally, we discuss (c) extensions to visualization tools in the Wikidata context for data exploration of the generated scholarly data. Our study focuses on data from 105 Semantic Web-related conferences and extends/adds more than 6000 entities in Wikidata. It is important to note that the method can be more generally applicable beyond Semantic Web-related conferences for enhancing Wikidata's utility as a comprehensive scholarly resource. Source Repository: https://github.com/scholarly-wikidata/ DOI: https://doi.org/10.5281/zenodo.10989709 License: Creative Commons CC0 (Data), MIT (Code)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16187v3">No Free Lunch in LLM Watermarking: Trade-offs in Watermarking Design Choices</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      Advances in generative models have made it possible for AI-generated text, code, and images to mirror human-generated content in many applications. Watermarking, a technique that aims to embed information in the output of a model to verify its source, is useful for mitigating the misuse of such AI-generated content. However, we show that common design choices in LLM watermarking schemes make the resulting systems surprisingly susceptible to attack -- leading to fundamental trade-offs in robustness, utility, and usability. To navigate these trade-offs, we rigorously study a set of simple yet effective attacks on common watermarking systems, and propose guidelines and defenses for LLM watermarking in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08640v1">Towards Secure Intelligent O-RAN Architecture: Vulnerabilities, Threats and Promising Technical Solutions using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The evolution of wireless communication systems will be fundamentally impacted by an open radio access network (O-RAN), a new concept defining an intelligent architecture with enhanced flexibility, openness, and the ability to slice services more efficiently. For all its promises, and like any technological advancement, O-RAN is not without risks that need to be carefully assessed and properly addressed to accelerate its wide adoption in future mobile networks. In this paper, we present an in-depth security analysis of the O-RAN architecture, discussing the potential threats that may arise in the different O-RAN architecture layers and their impact on the Confidentiality, Integrity, and Availability (CIA) triad. We also promote the potential of zero trust, Moving Target Defense (MTD), blockchain, and large language models(LLM) technologies in fortifying O-RAN's security posture. Furthermore, we numerically demonstrate the effectiveness of MTD in empowering robust deep reinforcement learning methods for dynamic network slice admission control in the O-RAN architecture. Moreover, we examine the effect of explainable AI (XAI) based on LLMs in securing the system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15736v2">General LLMs as Instructors for Domain-Specific LLMs: A Sequential Fusion Method to Integrate Extraction and Editing</a></div>
    <div class="paper-meta">
      📅 2024-11-13
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      The substantial interest in updating Large Language Models (LLMs) without retraining from scratch is accompanied by several challenges. This is particularly true when updating LLMs with datasets that necessitate domain-expert reasoning across extensive texts, despite limited samples. We termed the scenario as the Few-Shot Domain-Expert Reasoning for Updating LLMs (FDoR-UL). Traditional methods such as Low-Rank Adaptation (LoRA) and Retrieval Augmented Generation (RAG) are inadequate for addressing this critical issue, particularly evident in our exploration of a specific medical dataset that epitomizes the distinct needs of FDoR-UL. To tackle this challenge, we introduce a Sequential Fusion method to integrate knowledge from complex contexts into LLMs. This method employs a two-stage framework: initially leveraging general LLMs to perform relation extraction for knowledge acquisition from complex texts, followed by updating domain-specific LLMs through Knowledge Editing (KE). Employing our method, domain-specific LLMs achieved a 71.7% accuracy (an average gain of 39.1%) in question-answering tasks. Furthermore, we expanded our evaluation to a novel economics-management dataset we developed, where our method achieved a 75.0% accuracy (an average gain of 45.0%). These findings underscore the effectiveness and flexibility of our approach in FDoR-UL across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.08574v1">Practitioners' Discussions on Building LLM-based Applications for Production</a></div>
    <div class="paper-meta">
      📅 2024-11-13
    </div>
    <details class="paper-abstract">
      \textit{Background}: Large language models (LLMs) have become a paramount interest of researchers and practitioners alike, yet a comprehensive overview of key considerations for those developing LLM-based systems is lacking. This study addresses this gap by collecting and mapping the topics practitioners discuss online, offering practical insights into where priorities lie in developing LLM-based applications. \textit{Method}: We collected 189 videos from 2022 to 2024 from practitioners actively developing such systems and discussing various aspects they encounter during development and deployment of LLMs in production. We analyzed the transcripts using BERTopic, then manually sorted and merged the generated topics into themes, leading to a total of 20 topics in 8 themes. \textit{Results}: The most prevalent topics fall within the theme Design \& Architecture, with a strong focus on retrieval-augmented generation (RAG) systems. Other frequently discussed topics include model capabilities and enhancement techniques (e.g., fine-tuning, prompt engineering), infrastructure and tooling, and risks and ethical challenges. \textit{Implications}: Our results highlight current discussions and challenges in deploying LLMs in production. This way, we provide a systematic overview of key aspects practitioners should be aware of when developing LLM-based applications. We further pale off topics of interest for academics where further research is needed.
    </details>
</div>
