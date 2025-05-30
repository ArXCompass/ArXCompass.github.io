# llm - 2025_05

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
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15738v1">Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15710v1">Advancing LLM Safe Alignment with Safety Representation Ranking</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has demonstrated milestone success in a variety of tasks, yet their potential for generating harmful content has raised significant safety concerns. Existing safety evaluation approaches typically operate directly on textual responses, overlooking the rich information embedded in the model's internal representations. In this paper, we propose Safety Representation Ranking (SRR), a listwise ranking framework that selects safe responses using hidden states from the LLM itself. SRR encodes both instructions and candidate completions using intermediate transformer representations and ranks candidates via a lightweight similarity-based scorer. Our approach directly leverages internal model states and supervision at the list level to capture subtle safety signals. Experiments across multiple benchmarks show that SRR significantly improves robustness to adversarial prompts. Our code will be available upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15683v1">A Federated Splitting Framework for LLMs: Security, Efficiency, and Adaptability</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Private data is typically larger and of higher quality than public data, offering great potential to improve LLM. However, its scattered distribution across data silos and the high computational demands of LLMs limit their deployment in federated environments. To address this, the transformer-based split learning model has emerged, offloading most model parameters to the server while retaining only the embedding and output layers on clients to ensure privacy. However, it still faces significant challenges in security, efficiency, and adaptability: 1) embedding gradients are vulnerable to attacks, leading to reverse engineering of private data; 2) the autoregressive nature of LLMs means that federated split learning can only train and infer sequentially, causing high communication overhead; 3) fixed partition points lack adaptability to downstream tasks. In this paper, we introduce FL-LLaMA, a secure, efficient, and adaptive federated split framework based on LLaMA2. First, we place some input and output blocks on the local client and inject Gaussian noise into forward-pass hidden states, enabling secure end-to-end propagation. Second, we employ client-batch and server-hierarchical strategies to achieve parallel training, along with attention-mask compression and KV cache mechanisms to accelerate inference, reducing communication costs effectively. Third, we allow users to dynamically adjust the partition points for input/output blocks based on specific task requirements and hardware limitations. Experiments on NLU, summarization and conversational QA tasks show that FL-LLaMA maintains performance comparable to centralized LLaMA2, and achieves up to 2x train speedups and 8x inference speedups. Further analysis of privacy attacks and different partition points also demonstrates the effectiveness of FL-LLaMA in security and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17216v2">Intermediate Languages Matter: Formal Choice Drives Neurosymbolic LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve astonishing results on a wide range of tasks. However, their formal reasoning ability still lags behind. A promising approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators from natural to formal languages and symbolic solvers for deriving correct results. Still, it remains unclear what the contributing factors to the success of Neurosymbolic LLM reasoning are. This paper shows that one important factor is the choice of the formal language. By comparing 4 formal languages on 3 datasets over 6 LLMs, we show that the choice of formal language affects both the syntactic and the semantic reasoning capability. Thereby, we introduce the intermediate language challenge, which is the challenge of picking a suitable formal language for neurosymbolic reasoning. Further, we compare the effects of using different in-context-learning examples in an ablation study. We conclude that on average, context-aware encodings help LLMs to reason, while there is no apparent effect of using comments or markdown syntax.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10900v2">Explain What You Mean: Intent Augmented Knowledge Graph Recommender Built With An LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Interaction sparsity is a long-standing challenge in recommendation systems. Sparsity manifests in environments with disproportional cardinality of groupings of entities, such as users and products in an online marketplace. It is also found for newly introduced entities, described as the cold-start problem. Recent efforts to mitigate this issue either enrich the connectivity data by incorporating social networks or external knowledge graphs, or fine-tune LLMs into interaction augmenters or next-item recommenders. However, these techniques tend to be resource demanding, requiring high computational power. They also have several limitations, including data availability, low quality, or synthetic noise issues. In this work, we propose LLM-based Intent Knowledge Graph Recommender (IKGR), a novel framework that leverages retrieval-augmented generation and an encoding approach to construct and densify a knowledge graph. IKGR leverages latent user-item affinities from an interaction knowledge graph and further densifies it through mutual intent connectivity. This addresses sparsity issues and allows the model to make intent-grounded recommendations with an interpretable embedding translation layer. Through extensive experiments on real-world datasets, we demonstrate that IKGR overcomes knowledge gaps and achieves substantial gains over state-of-the-art baselines on both publicly available and our internal recommendation datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15656v1">Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15623v1">Can LLMs $\textit{understand}$ Math? -- Exploring the Pitfalls in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate considerable potential in various natural language tasks but face significant challenges in mathematical reasoning, particularly in executing precise, multi-step logic. However, current evaluation frameworks judge their performance solely based on accuracy, which only accounts for the final answer. This study explores these pitfalls by employing a novel evaluation framework. We propose an evaluation metric called the MAPLE score, which holistically quantifies reasoning misalignment by integrating error rates, redundancy, and validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15607v1">From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 David Dinucu-Jianu and Jakub Macina contributed equally. Code available: https://github.com/eth-lre/PedagogicalRL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10259v3">SpecOffload: Unlocking Latent GPU Capacity for LLM Inference on Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Efficient LLM inference on resource-constrained devices presents significant challenges in compute and memory utilization. Due to limited GPU memory, existing systems offload model weights to CPU memory, incurring substantial I/O overhead between the CPU and GPU. This leads to two major inefficiencies: (1) GPU cores are underutilized, often remaining idle while waiting for data to be loaded; and (2) GPU memory has low impact on performance, as reducing its capacity has minimal effect on overall throughput.In this paper, we propose SpecOffload, a high-throughput inference engine that embeds speculative decoding into offloading. Our key idea is to unlock latent GPU resources for storing and executing a draft model used for speculative decoding, thus accelerating inference at near-zero additional cost. To support this, we carefully orchestrate the interleaved execution of target and draft models in speculative decoding within the offloading pipeline, and propose a planner to manage tensor placement and select optimal parameters. Compared to the best baseline, SpecOffload improves GPU core utilization by 4.49x and boosts inference throughput by 2.54x. Our code is available at https://github.com/MobiSense/SpecOffload-public .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15596v1">Exploring LLM-Generated Feedback for Economics Essays: How Teaching Assistants Evaluate and Envision Its Use</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 To be published in AIED'2025: In Proceedings of the 26th International Conference on Artificial Intelligence in Education. The system prompt and example feedback can be found through http://github.com/UM-Lifelong-Learning-Lab/AIED2025-Exploring-LLM-Generated-Feedback-for-Economics-Essay
    </div>
    <details class="paper-abstract">
      This project examines the prospect of using AI-generated feedback as suggestions to expedite and enhance human instructors' feedback provision. In particular, we focus on understanding the teaching assistants' perspectives on the quality of AI-generated feedback and how they may or may not utilize AI feedback in their own workflows. We situate our work in a foundational college Economics class, which has frequent short essay assignments. We developed an LLM-powered feedback engine that generates feedback on students' essays based on grading rubrics used by the teaching assistants (TAs). To ensure that TAs can meaningfully critique and engage with the AI feedback, we had them complete their regular grading jobs. For a randomly selected set of essays that they had graded, we used our feedback engine to generate feedback and displayed the feedback as in-text comments in a Word document. We then performed think-aloud studies with 5 TAs over 20 1-hour sessions to have them evaluate the AI feedback, contrast the AI feedback with their handwritten feedback, and share how they envision using the AI feedback if they were offered as suggestions. The study highlights the importance of providing detailed rubrics for AI to generate high-quality feedback for knowledge-intensive essays. TAs considered that using AI feedback as suggestions during their grading could expedite grading, enhance consistency, and improve overall feedback quality. We discuss the importance of decomposing the feedback generation task into steps and presenting intermediate results, in order for TAs to use the AI feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14336v2">Scaling and Enhancing LLM-based AVSR: A Sparse Mixture of Projectors Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Interspeech 2025
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) enhances robustness in noisy environments by integrating visual cues. While recent advances integrate Large Language Models (LLMs) into AVSR, their high computational cost hinders deployment in resource-constrained settings. To address this, we propose Llama-SMoP, an efficient Multimodal LLM that employs a Sparse Mixture of Projectors (SMoP) module to scale model capacity without increasing inference costs. By incorporating sparsely-gated mixture-of-experts (MoE) projectors, Llama-SMoP enables the use of smaller LLMs while maintaining strong performance. We explore three SMoP configurations and show that Llama-SMoP DEDR (Disjoint-Experts, Disjoint-Routers), which uses modality-specific routers and experts, achieves superior performance on ASR, VSR, and AVSR tasks. Ablation studies confirm its effectiveness in expert activation, scalability, and noise robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v4">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.We conduct an repository for our benchmark at https://github.com/zjq0455/PTQ_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17644v4">BurstGPT: A Real-world Workload Dataset to Optimize LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Serving systems for Large Language Models (LLMs) are often optimized to improve quality of service (QoS) and throughput. However, due to the lack of open-source LLM serving workloads, these systems are frequently evaluated under unrealistic workload assumptions. Consequently, performance may degrade when systems are deployed in real-world scenarios. This work presents BurstGPT, an LLM serving workload with 10.31 million traces from regional Azure OpenAI GPT services over 213 days. BurstGPT captures LLM serving characteristics from user, model and system perspectives: (1) User request concurrency: burstiness variations of requests in Azure OpenAI GPT services, revealing diversified concurrency patterns in different services and model types. (2) User conversation patterns: counts and intervals within conversations for service optimizations. (3) Model response lengths: auto-regressive serving processes of GPT models, showing statistical relations between requests and their responses. (4) System response failures: failures of conversation and API services, showing intensive resource needs and limited availability of LLM services in Azure. The details of the characteristics can serve multiple purposes in LLM serving optimizations, such as system evaluation and trace provisioning. In our demo evaluation with BurstGPT, frequent variations in BurstGPT reveal declines in efficiency, stability, or reliability in realistic LLM serving. We identify that the generalization of KV cache management, scheduling and disaggregation optimizations can be improved under realistic workload evaluations. BurstGPT is publicly available now at https://github.com/HPMLL/BurstGPT and is widely used to develop prototypes of LLM serving frameworks in the industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15524v1">Evaluate Bias without Manual Test Sets: A Concept Representation Perspective for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Bias in Large Language Models (LLMs) significantly undermines their reliability and fairness. We focus on a common form of bias: when two reference concepts in the model's concept space, such as sentiment polarities (e.g., "positive" and "negative"), are asymmetrically correlated with a third, target concept, such as a reviewing aspect, the model exhibits unintended bias. For instance, the understanding of "food" should not skew toward any particular sentiment. Existing bias evaluation methods assess behavioral differences of LLMs by constructing labeled data for different social groups and measuring model responses across them, a process that requires substantial human effort and captures only a limited set of social concepts. To overcome these limitations, we propose BiasLens, a test-set-free bias analysis framework based on the structure of the model's vector space. BiasLens combines Concept Activation Vectors (CAVs) with Sparse Autoencoders (SAEs) to extract interpretable concept representations, and quantifies bias by measuring the variation in representational similarity between the target concept and each of the reference concepts. Even without labeled data, BiasLens shows strong agreement with traditional bias evaluation metrics (Spearman correlation r > 0.85). Moreover, BiasLens reveals forms of bias that are difficult to detect using existing methods. For example, in simulated clinical scenarios, a patient's insurance status can cause the LLM to produce biased diagnostic assessments. Overall, BiasLens offers a scalable, interpretable, and efficient paradigm for bias discovery, paving the way for improving fairness and transparency in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15501v1">Protoknowledge Shapes Behaviour of LLMs in Downstream Tasks: Memorization and Generalization with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      We introduce the concept of protoknowledge to formalize and measure how sequences of tokens encoding Knowledge Graphs are internalized during pretraining and utilized at inference time by Large Language Models (LLMs). Indeed, LLMs have demonstrated the ability to memorize vast amounts of token sequences during pretraining, and a central open question is how they leverage this memorization as reusable knowledge through generalization. We then categorize protoknowledge into lexical, hierarchical, and topological forms, varying on the type of knowledge that needs to be activated. We measure protoknowledge through Knowledge Activation Tasks (KATs), analyzing its general properties such as semantic bias. We then investigate the impact of protoknowledge on Text-to-SPARQL performance by varying prompting strategies depending on input conditions. To this end, we adopt a novel analysis framework that assesses whether model predictions align with the successful activation of the relevant protoknowledge for each query. This methodology provides a practical tool to explore Semantic-Level Data Contamination and serves as an effective strategy for Closed-Pretraining models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15480v1">KaFT: Knowledge-aware Fine-tuning for Boosting LLMs' Domain-specific Question-Answering Performance</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted to ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common approach to improve the domain-specific question-answering (QA) performance of large language models (LLMs). However, recent literature reveals that due to the conflicts between LLMs' internal knowledge and the context knowledge of training data, vanilla SFT using the full QA training set is usually suboptimal. In this paper, we first design a query diversification strategy for robust conflict detection and then conduct a series of experiments to analyze the impact of knowledge conflict. We find that 1) training samples with varied conflicts contribute differently, where SFT on the data with large conflicts leads to catastrophic performance drops; 2) compared to directly filtering out the conflict data, appropriately applying the conflict data would be more beneficial. Motivated by this, we propose a simple-yet-effective Knowledge-aware Fine-tuning (namely KaFT) approach to effectively boost LLMs' performance. The core of KaFT is to adapt the training weight by assigning different rewards for different training samples according to conflict level. Extensive experiments show that KaFT brings consistent and significant improvements across four LLMs. More analyses prove that KaFT effectively improves the model generalization and alleviates the hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15444v1">Single LLM, Multiple Roles: A Unified Retrieval-Augmented Generation Framework Using Role-Specific Token Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Existing studies have optimized retrieval-augmented generation (RAG) across various sub-tasks, such as query understanding and retrieval refinement, but integrating these optimizations into a unified framework remains challenging. To tackle this problem, this work proposes RoleRAG, a unified RAG framework that achieves efficient multi-task processing through role-specific token optimization. RoleRAG comprises six modules, each handling a specific sub-task within the RAG process. Additionally, we introduce a query graph to represent the decomposition of the query, which can be dynamically resolved according to the decomposing state. All modules are driven by the same underlying LLM, distinguished by task-specific role tokens that are individually optimized. This design allows RoleRAG to dynamically activate different modules within a single LLM instance, thereby streamlining deployment and reducing resource consumption. Experimental results on five open-domain question-answering datasets demonstrate the effectiveness, generalizability, and flexibility of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15433v1">Set-LLM: A Permutation-Invariant LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) demonstrate impressive capabilities across numerous applications, their robustness remains a critical concern. This paper is motivated by a specific vulnerability: the order sensitivity of LLMs. This vulnerability manifests itself as the order bias observed when LLMs decide between possible options (for example, a preference for the first option) and the tendency of LLMs to provide different answers when options are reordered. The use cases for this scenario extend beyond the classical case of multiple-choice question answering to the use of LLMs as automated evaluators in AI pipelines, comparing output generated by different models. We introduce Set-LLM, a novel architectural adaptation for pretrained LLMs that enables the processing of mixed set-text inputs with permutation invariance guarantees. The adaptations involve a new attention mask and new positional encodings specifically designed for sets. We provide a theoretical proof of invariance and demonstrate through experiments that Set-LLM can be trained effectively, achieving comparable or improved performance and maintaining the runtime of the original model, while eliminating order sensitivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15426v1">NeoN: A Tool for Automated Detection, Linguistic and LLM-Driven Analysis of Neologisms in Polish</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 15 pages, this is an extended version of a paper accepted for the 25th International Conference on Computational Science (ICCS), 7-9 July 2025
    </div>
    <details class="paper-abstract">
      NeoN, a tool for detecting and analyzing Polish neologisms. Unlike traditional dictionary-based methods requiring extensive manual review, NeoN combines reference corpora, Polish-specific linguistic filters, an LLM-driven precision-boosting filter, and daily RSS monitoring in a multi-layered pipeline. The system uses context-aware lemmatization, frequency analysis, and orthographic normalization to extract candidate neologisms while consolidating inflectional variants. Researchers can verify candidates through an intuitive interface with visualizations and filtering controls. An integrated LLM module automatically generates definitions and categorizes neologisms by domain and sentiment. Evaluations show NeoN maintains high accuracy while significantly reducing manual effort, providing an accessible solution for tracking lexical innovation in Polish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15422v1">Trends and Challenges in Authorship Analysis: A Review of ML, DL, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 25 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Authorship analysis plays an important role in diverse domains, including forensic linguistics, academia, cybersecurity, and digital content authentication. This paper presents a systematic literature review on two key sub-tasks of authorship analysis; Author Attribution and Author Verification. The review explores SOTA methodologies, ranging from traditional ML approaches to DL models and LLMs, highlighting their evolution, strengths, and limitations, based on studies conducted from 2015 to 2024. Key contributions include a comprehensive analysis of methods, techniques, their corresponding feature extraction techniques, datasets used, and emerging challenges in authorship analysis. The study highlights critical research gaps, particularly in low-resource language processing, multilingual adaptation, cross-domain generalization, and AI-generated text detection. This review aims to help researchers by giving an overview of the latest trends and challenges in authorship analysis. It also points out possible areas for future study. The goal is to support the development of better, more reliable, and accurate authorship analysis system in diverse textual domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15410v1">ClickSight: Interpreting Student Clickstreams to Reveal Insights on Learning Strategies via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted in Latebreaking results track in AIED 2025(26th International Conference on Artificial Intelligence in Education JULY 22-26, 2025 PALERMO, ITALY)
    </div>
    <details class="paper-abstract">
      Clickstream data from digital learning environments offer valuable insights into students' learning behaviors, but are challenging to interpret due to their high dimensionality and granularity. Prior approaches have relied mainly on handcrafted features, expert labeling, clustering, or supervised models, therefore often lacking generalizability and scalability. In this work, we introduce ClickSight, an in-context Large Language Model (LLM)-based pipeline that interprets student clickstreams to reveal their learning strategies. ClickSight takes raw clickstreams and a list of learning strategies as input and generates textual interpretations of students' behaviors during interaction. We evaluate four different prompting strategies and investigate the impact of self-refinement on interpretation quality. Our evaluation spans two open-ended learning environments and uses a rubric-based domain-expert evaluation. Results show that while LLMs can reasonably interpret learning strategies from clickstreams, interpretation quality varies by prompting strategy, and self-refinement offers limited improvement. ClickSight demonstrates the potential of LLMs to generate theory-driven insights from educational interaction data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15392v1">An Empirical Study of the Anchoring Effect in LLMs: Existence, Mechanism, and Potential Mitigations</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) like ChatGPT has advanced natural language processing, yet concerns about cognitive biases are growing. In this paper, we investigate the anchoring effect, a cognitive bias where the mind relies heavily on the first information as anchors to make affected judgments. We explore whether LLMs are affected by anchoring, the underlying mechanisms, and potential mitigation strategies. To facilitate studies at scale on the anchoring effect, we introduce a new dataset, SynAnchors. Combining refined evaluation metrics, we benchmark current widely used LLMs. Our findings show that LLMs' anchoring bias exists commonly with shallow-layer acting and is not eliminated by conventional strategies, while reasoning can offer some mitigation. This recontextualization via cognitive psychology urges that LLM evaluations focus not on standard benchmarks or over-optimized robustness tests, but on cognitive-bias-aware trustworthy evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15365v1">AI vs. Human Judgment of Content Moderation: LLM-as-a-Judge and Ethics-Based Response Refusals</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in high-stakes settings, their ability to refuse ethically sensitive prompts-such as those involving hate speech or illegal activities-has become central to content moderation and responsible AI practices. While refusal responses can be viewed as evidence of ethical alignment and safety-conscious behavior, recent research suggests that users may perceive them negatively. At the same time, automated assessments of model outputs are playing a growing role in both evaluation and training. In particular, LLM-as-a-Judge frameworks-in which one model is used to evaluate the output of another-are now widely adopted to guide benchmarking and fine-tuning. This paper examines whether such model-based evaluators assess refusal responses differently than human users. Drawing on data from Chatbot Arena and judgments from two AI judges (GPT-4o and Llama 3 70B), we compare how different types of refusals are rated. We distinguish ethical refusals, which explicitly cite safety or normative concerns (e.g., "I can't help with that because it may be harmful"), and technical refusals, which reflect system limitations (e.g., "I can't answer because I lack real-time data"). We find that LLM-as-a-Judge systems evaluate ethical refusals significantly more favorably than human users, a divergence not observed for technical refusals. We refer to this divergence as a moderation bias-a systematic tendency for model-based evaluators to reward refusal behaviors more than human users do. This raises broader questions about transparency, value alignment, and the normative assumptions embedded in automated evaluation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v2">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 41 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01941v2">Can LLMs Maintain Fundamental Abilities under KV Cache Compression?</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      This paper investigates an underexplored challenge in large language models (LLMs): the impact of KV cache compression methods on LLMs' fundamental capabilities. Although existing methods achieve impressive compression ratios on long-context benchmarks, their effects on core model capabilities remain understudied. We present a comprehensive benchmark KVFundaBench to systematically evaluate the effects of KV cache compression across diverse fundamental LLM capabilities, spanning world knowledge, commonsense reasoning, arithmetic reasoning, code generation, safety, and long-context understanding and generation.Our analysis reveals serval key findings: (1) \textit{Task-Dependent Degradation}; (2) \textit{Model-Type Robustness} (3) \textit{Prompt Length Vulnerability}; (4) \textit{Chunk-Level Superiority}; (5) \textit{Prompt-Gain Sensitivity}; (6) \textit{Long-Context Generation Sensitivity}. Based on our analysis of attention patterns and cross-task compression performance, we propose ShotKV, a novel compression approach that distinctly handles prefill and decoding phases while maintaining shot-level semantic coherence. Empirical results show that ShotKV achieves $9\%$-$18\%$ performance improvements on long-context generation tasks under aggressive compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04964v3">Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15347v1">FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15337v1">Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15323v1">Improving LLM First-Token Predictions in Multiple-Choice Question Answering via Prefilling Attack</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 13 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly evaluated on multiple-choice question answering (MCQA) tasks using *first-token probability* (FTP), which selects the answer option whose initial token has the highest likelihood. While efficient, FTP can be fragile: models may assign high probability to unrelated tokens (*misalignment*) or use a valid token merely as part of a generic preamble rather than as a clear answer choice (*misinterpretation*), undermining the reliability of symbolic evaluation. We propose a simple solution: the *prefilling attack*, a structured natural-language prefix (e.g., "*The correct option is:*") prepended to the model output. Originally explored in AI safety, we repurpose prefilling to steer the model to respond with a clean, valid option, without modifying its parameters. Empirically, the FTP with prefilling strategy substantially improves accuracy, calibration, and output consistency across a broad set of LLMs and MCQA benchmarks. It outperforms standard FTP and often matches the performance of open-ended generation approaches that require full decoding and external classifiers, while being significantly more efficient. Our findings suggest that prefilling is a simple, robust, and low-cost method to enhance the reliability of FTP-based evaluation in multiple-choice settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15311v1">Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v3">Automate Strategy Finding with LLM in Quant Investment</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      We present a novel three-stage framework leveraging Large Language Models (LLMs) within a risk-aware multi-agent system for automate strategy finding in quantitative finance. Our approach addresses the brittleness of traditional deep learning models in financial applications by: employing prompt-engineered LLMs to generate executable alpha factor candidates across diverse financial data, implementing multimodal agent-based evaluation that filters factors based on market status, predictive quality while maintaining category balance, and deploying dynamic weight optimization that adapts to market conditions. Experimental results demonstrate the robust performance of the strategy in Chinese & US market regimes compared to established benchmarks. Our work extends LLMs capabilities to quantitative trading, providing a scalable architecture for financial signal extraction and portfolio construction. The overall framework significantly outperforms all benchmarks with 53.17% cumulative return on SSE50 (Jan 2023 to Jan 2024), demonstrating superior risk-adjusted performance and downside protection on the market.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14350v3">An Empirical Study of LLM Reasoning Ability Under Strict Output Length Constraint</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent work has demonstrated the remarkable potential of Large Language Models (LLMs) in test-time scaling. By making models think before answering, they are able to achieve much higher accuracy with extra inference computation. However, in many real-world scenarios, models are used under time constraints, where an answer should be given within a certain output length. It is unclear whether and how the reasoning ability of different LLMs remain effective under strict constraints. We take a first look at this problem by conducting an in-depth empirical study. Specifically, we test 30 LLMs on common reasoning datasets under a wide range of output length budgets, and we analyze the correlation between the inference accuracy and various properties including model type, model size, prompt style, etc. We also consider the mappings between token budgets and actual on-device latency budgets. The results have demonstrated several interesting findings regarding the budget-aware LLM reasoning ability that differ from the unconstrained situation, e.g. the optimal choices of either model size or prompt style change under different budgets. These findings offer timely evaluation to this area and practical guidance for users to deploy LLMs under real-world latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15257v1">When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 26 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-source LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively decoupled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training such as supervised fine-tuning or reinforcement learning, our training-free ablation achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15240v1">Generalised Probabilistic Modelling and Improved Uncertainty Estimation in Comparative LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 To appear in UAI 2025
    </div>
    <details class="paper-abstract">
      This paper explores generalised probabilistic modelling and uncertainty estimation in comparative LLM-as-a-judge frameworks. We show that existing Product-of-Experts methods are specific cases of a broader framework, enabling diverse modelling options. Furthermore, we propose improved uncertainty estimates for individual comparisons, enabling more efficient selection and achieving strong performance with fewer evaluations. We also introduce a method for estimating overall ranking uncertainty. Finally, we demonstrate that combining absolute and comparative scoring improves performance. Experiments show that the specific expert model has a limited impact on final rankings but our proposed uncertainty estimates, especially the probability of reordering, significantly improve the efficiency of systems reducing the number of needed comparisons by ~50%. Furthermore, ranking-level uncertainty metrics can be used to identify low-performing predictions, where the nature of the probabilistic model has a notable impact on the quality of the overall uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15229v1">Multilingual Prompting for Improving LLM Generation Diversity</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to lack cultural representation and overall diversity in their generations, from expressing opinions to answering factual questions. To mitigate this problem, we propose multilingual prompting: a prompting method which generates several variations of a base prompt with added cultural and linguistic cues from several cultures, generates responses, and then combines the results. Building on evidence that LLMs have language-specific knowledge, multilingual prompting seeks to increase diversity by activating a broader range of cultural knowledge embedded in model training data. Through experiments across multiple models (GPT-4o, GPT-4o-mini, LLaMA 70B, and LLaMA 8B), we show that multilingual prompting consistently outperforms existing diversity-enhancing techniques such as high-temperature sampling, step-by-step recall, and personas prompting. Further analyses show that the benefits of multilingual prompting vary with language resource level and model size, and that aligning the prompting language with the cultural cues reduces hallucination about culturally-specific information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05179v2">Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled strong reasoning capabilities through Chain-of-Thought (CoT) prompting, which elicits step-by-step problem solving, but often at the cost of excessive verbosity in intermediate outputs, leading to increased computational overhead. We propose Sketch-of-Thought (SoT), a prompting framework that integrates cognitively inspired reasoning paradigms with linguistic constraints to reduce token usage while preserving reasoning accuracy. SoT is designed as a flexible, modular approach and is instantiated with three paradigms--Conceptual Chaining, Chunked Symbolism, and Expert Lexicons--each tailored to distinct reasoning tasks and selected dynamically at test-time by a lightweight routing model. Across 15 reasoning datasets spanning multiple domains, languages, and modalities, SoT achieves token reductions of up to 78% with minimal accuracy loss. In tasks such as mathematical and multi-hop reasoning, it even improves accuracy while shortening outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02145v3">From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 New version of the paper
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14552v2">KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) underscore the need for more comprehensive evaluation methods to accurately assess their reasoning capabilities. Existing benchmarks are often domain-specific and thus cannot fully capture an LLM's general reasoning potential. To address this limitation, we introduce the Knowledge Orthogonal Reasoning Gymnasium (KORGym), a dynamic evaluation platform inspired by KOR-Bench and Gymnasium. KORGym offers over fifty games in either textual or visual formats and supports interactive, multi-turn assessments with reinforcement learning scenarios. Using KORGym, we conduct extensive experiments on 19 LLMs and 8 VLMs, revealing consistent reasoning patterns within model families and demonstrating the superior performance of closed-source models. Further analysis examines the effects of modality, reasoning strategies, reinforcement learning techniques, and response length on model performance. We expect KORGym to become a valuable resource for advancing LLM reasoning research and developing evaluation methodologies suited to complex, interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09049v3">Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Discovering customer intentions in dialogue conversations is crucial for automated service agents. However, existing intent clustering methods often fail to align with human perceptions due to a heavy reliance on embedding distance metrics and a tendency to overlook underlying semantic structures. This paper proposes an LLM-in-the-loop (LLM-ITL) intent clustering framework, integrating the semantic understanding capabilities of LLMs into conventional clustering algorithms. Specifically, this paper (1) investigates the effectiveness of fine-tuned LLMs in semantic coherence evaluation and intent cluster naming, achieving over 95% accuracy aligned with human judgments; (2) designs an LLM-ITL framework that facilitates the iterative discovery of coherent intent clusters and the optimal number of clusters; and (3) proposes context-aware techniques tailored for customer service dialogue. As existing English benchmarks offer limited semantic diversity and intent groups, we introduce a comprehensive Chinese dialogue intent dataset, comprising over 100k real customer service calls and 1,507 human-annotated intent clusters. The proposed approaches significantly outperform LLM-guided baselines, achieving notable enhancements in clustering quality and lower computational cost. Combined with several best practices, our findings highlight the potential of LLM-in-the-loop techniques for scalable and human-aligned intent clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15182v1">ReflAct: World-Grounded Decision Making in LLM Agents via Goal-State Reflection</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in LLM agents have largely built on reasoning backbones like ReAct, which interleave thought and action in complex environments. However, ReAct often produces ungrounded or incoherent reasoning steps, leading to misalignment between the agent's actual state and goal. Our analysis finds that this stems from ReAct's inability to maintain consistent internal beliefs and goal alignment, causing compounding errors and hallucinations. To address this, we introduce ReflAct, a novel backbone that shifts reasoning from merely planning next actions to continuously reflecting on the agent's state relative to its goal. By explicitly grounding decisions in states and enforcing ongoing goal alignment, ReflAct dramatically improves strategic reliability. This design delivers substantial empirical gains: ReflAct surpasses ReAct by 27.7% on average, achieving a 93.3% success rate in ALFWorld. Notably, ReflAct even outperforms ReAct with added enhancement modules (e.g., Reflexion, WKM), showing that strengthening the core reasoning backbone is key to reliable agent performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02009v2">Towards Safer Pretraining: Analyzing and Filtering Harmful Content in Webscale datasets for Responsible LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 10 pages, 5 figures. Accepted at the International Joint Conferences on Artificial Intelligence IJCAI 2025 (main track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become integral to various real-world applications, leveraging massive, web-sourced datasets like Common Crawl, C4, and FineWeb for pretraining. While these datasets provide linguistic data essential for high-quality natural language generation, they often contain harmful content, such as hate speech, misinformation, and biased narratives. Training LLMs on such unfiltered data risks perpetuating toxic behaviors, spreading misinformation, and amplifying societal biases which can undermine trust in LLM-driven applications and raise ethical concerns about their use. This paper presents a large-scale analysis of inappropriate content across these datasets, offering a comprehensive taxonomy that categorizes harmful webpages into Topical and Toxic based on their intent. We also introduce a prompt evaluation dataset, a high-accuracy Topical and Toxic Prompt (TTP), and a transformer-based model (HarmFormer) for harmful content filtering. Additionally, we create a new multi-harm open-ended toxicity benchmark (HAVOC) and provide crucial insights into how models respond to adversarial toxic inputs. We share TTP, TTP-Eval, HAVOC and a sample of C4 inferenced on HarmFormer. Our work offers insights into ensuring safer LLM pretraining and serves as a resource for Responsible AI (RAI) compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15154v1">Prolonged Reasoning Is Not All You Need: Certainty-Based Adaptive Routing for Efficient LLM/MLLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advancements in reasoning have significantly enhanced the capabilities of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) across diverse tasks. However, excessive reliance on chain-of-thought (CoT) reasoning can impair model performance and brings unnecessarily lengthened outputs, reducing efficiency. Our work reveals that prolonged reasoning does not universally improve accuracy and even degrade performance on simpler tasks. To address this, we propose Certainty-based Adaptive Reasoning (CAR), a novel framework that dynamically switches between short answers and long-form reasoning based on the model perplexity. CAR first generates a short answer and evaluates its perplexity, triggering reasoning only when the model exhibits low confidence (i.e., high perplexity). Experiments across diverse multimodal VQA/KIE benchmarks and text reasoning datasets show that CAR outperforms both short-answer and long-form reasoning approaches, striking an optimal balance between accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04295v3">Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant capability across various tasks, with their real-world effectiveness often driven by prompt design. While recent research has focused on optimizing prompt content, the role of prompt formatting, a critical but often overlooked dimension, has received limited systematic investigation. In this paper, we introduce Content-Format Integrated Prompt Optimization (CFPO), an innovative methodology that jointly optimizes both prompt content and formatting through an iterative refinement process. CFPO leverages natural language mutations to explore content variations and employs a dynamic format exploration strategy that systematically evaluates diverse format options. Our extensive evaluations across multiple tasks and open-source LLMs demonstrate that CFPO demonstrates measurable performance improvements compared to content-only optimization methods. This highlights the importance of integrated content-format optimization and offers a practical, model-agnostic approach to enhancing LLM performance. Code is available at https://github.com/HenryLau7/CFPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15146v1">lmgame-Bench: How Good are LLMs at Playing Games?</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Playing video games requires perception, memory, and planning, exactly the faculties modern large language model (LLM) agents are expected to master. We study the major challenges in using popular video games to evaluate modern LLMs and find that directly dropping LLMs into games cannot make an effective evaluation, for three reasons -- brittle vision perception, prompt sensitivity, and potential data contamination. We introduce lmgame-Bench to turn games into reliable evaluations. lmgame-Bench features a suite of platformer, puzzle, and narrative games delivered through a unified Gym-style API and paired with lightweight perception and memory scaffolds, and is designed to stabilize prompt variance and remove contamination. Across 13 leading models, we show lmgame-Bench is challenging while still separating models well. Correlation analysis shows that every game probes a unique blend of capabilities often tested in isolation elsewhere. More interestingly, performing reinforcement learning on a single game from lmgame-Bench transfers both to unseen games and to external planning tasks. Our evaluation code is available at https://github.com/lmgame-org/GamingAgent/lmgame-bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20302v2">A Multilingual, Culture-First Approach to Addressing Misgendering in LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Misgendering is the act of referring to someone by a gender that does not match their chosen identity. It marginalizes and undermines a person's sense of self, causing significant harm. English-based approaches have clear-cut approaches to avoiding misgendering, such as the use of the pronoun ``they''. However, other languages pose unique challenges due to both grammatical and cultural constructs. In this work we develop methodologies to assess and mitigate misgendering across 42 languages and dialects using a participatory-design approach to design effective and appropriate guardrails across all languages. We test these guardrails in a standard LLM-based application (meeting transcript summarization), where both the data generation and the annotation steps followed a human-in-the-loop approach. We find that the proposed guardrails are very effective in reducing misgendering rates across all languages in the summaries generated, and without incurring loss of quality. Our human-in-the-loop approach demonstrates a method to feasibly scale inclusive and responsible AI-based solutions across multiple languages and cultures. We release the guardrails and synthetic dataset encompassing 42 languages, along with human and LLM-judge evaluations, to encourage further research on this subject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15134v1">The Unreasonable Effectiveness of Entropy Minimization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Entropy minimization (EM) trains the model to concentrate even more probability mass on its most confident outputs. We show that this simple objective alone, without any labeled data, can substantially improve large language models' (LLMs) performance on challenging math, physics, and coding tasks. We explore three approaches: (1) EM-FT minimizes token-level entropy similarly to instruction finetuning, but on unlabeled outputs drawn from the model; (2) EM-RL: reinforcement learning with negative entropy as the only reward to maximize; (3) EM-INF: inference-time logit adjustment to reduce entropy without any training data or parameter updates. On Qwen-7B, EM-RL, without any labeled data, achieves comparable or better performance than strong RL baselines such as GRPO and RLOO that are trained on 60K labeled examples. Furthermore, EM-INF enables Qwen-32B to match or exceed the performance of proprietary models like GPT-4o, Claude 3 Opus, and Gemini 1.5 Pro on the challenging SciCode benchmark, while being 3x more efficient than self-consistency and sequential refinement. Our findings reveal that many pretrained LLMs possess previously underappreciated reasoning capabilities that can be effectively elicited through entropy minimization alone, without any labeled data or even any parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05849v2">AGENTFUZZER: Generic Black-Box Fuzzing for Indirect Prompt Injection against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11023v2">Beyond A Single AI Cluster: A Survey of Decentralized LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has revolutionized AI development, yet the resource demands beyond a single cluster or even datacenter, limiting accessibility to well-resourced organizations. Decentralized training has emerged as a promising paradigm to leverage dispersed resources across clusters, datacenters and regions, offering the potential to democratize LLM development for broader communities. As the first comprehensive exploration of this emerging field, we present decentralized LLM training as a resource-driven paradigm and categorize existing efforts into community-driven and organizational approaches. We further clarify this through: (1) a comparison with related paradigms, (2) a characterization of decentralized resources, and (3) a taxonomy of recent advancements. We also provide up-to-date case studies and outline future directions to advance research in decentralized LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v7">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15117v1">An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15107v1">StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at https://github.com/zxh20001117/StepSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v2">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15101v1">Cost-aware LLM-based Online Dataset Annotation</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled automated dataset labeling with minimal human supervision. While majority voting across multiple LLMs can improve label reliability by mitigating individual model biases, it incurs high computational costs due to repeated querying. In this work, we propose a novel online framework, Cost-aware Majority Voting (CaMVo), for efficient and accurate LLM-based dataset annotation. CaMVo adaptively selects a subset of LLMs for each data instance based on contextual embeddings, balancing confidence and cost without requiring pre-training or ground-truth labels. Leveraging a LinUCB-based selection mechanism and a Bayesian estimator over confidence scores, CaMVo estimates a lower bound on labeling accuracy for each LLM and aggregates responses through weighted majority voting. Our empirical evaluation on the MMLU and IMDB Movie Review datasets demonstrates that CaMVo achieves comparable or superior accuracy to full majority voting while significantly reducing labeling costs. This establishes CaMVo as a practical and robust solution for cost-efficient annotation in dynamic labeling environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03797v3">NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The resurgence of autonomous agents built using large language models (LLMs) to solve complex real-world tasks has brought increased focus on LLMs' fundamental ability of tool or function calling. At the core of these agents, an LLM must plan, execute, and respond using external tools, APIs, and custom functions. Research on tool calling has gathered momentum, but evaluation benchmarks and datasets representing the complexity of the tasks have lagged behind. In this work, we focus on one such complexity, nested sequencing, with the goal of extending existing benchmarks and evaluation. Specifically, we present NESTFUL, a benchmark to evaluate LLMs on nested sequences of API calls, i.e., sequences where the output of one API call is passed as input to a subsequent call. NESTFUL contains 1800+ nested sequences where all the function calls are executable. Experimental results on a variety of models show that the best-performing model (GPT-4o) achieves a full sequence match accuracy of 28% and a win-rate of 60%, necessitating a large scope for improvement in the nested sequencing aspect of function calling. Our analysis of these results provides possible future research directions for the community, in addition to a benchmark to track progress. We have released the NESTFUL dataset under the Apache 2.0 license at https://github.com/IBM/NESTFUL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15091v1">ThinkRec: Thinking-based recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: https://anonymous.4open.science/r/ThinkRec_LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13989v2">When LLMs meet open-world graph learning: a new perspective for unlabeled data uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have significantly advanced text-attributed graph (TAG) learning. However, existing methods inadequately handle data uncertainty in open-world scenarios, especially concerning limited labeling and unknown-class nodes. Prior solutions typically rely on isolated semantic or structural approaches for unknown-class rejection, lacking effective annotation pipelines. To address these limitations, we propose Open-world Graph Assistant (OGA), an LLM-based framework that combines adaptive label traceability, which integrates semantics and topology for unknown-class rejection, and a graph label annotator to enable model updates using newly annotated nodes. Comprehensive experiments demonstrate OGA's effectiveness and practicality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v2">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 Accepted by ACL2025(Findings)
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11772v2">LAMP: Extracting Locally Linear Decision Surfaces from LLM World Models</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      We introduce LAMP (Linear Attribution Mapping Probe), a method that shines light onto a black-box language model's decision surface and studies how reliably a model maps its stated reasons to its predictions through a locally linear model approximating the decision surface. LAMP treats the model's own self-reported explanations as a coordinate system and fits a locally linear surrogate that links those weights to the model's output. By doing so, it reveals which stated factors steer the model's decisions, and by how much. We apply LAMP to three tasks: sentiment analysis, controversial-topic detection, and safety-prompt auditing. Across these tasks, LAMP reveals that many LLMs exhibit locally linear decision landscapes. In addition, these surfaces correlate with human judgments on explanation quality and, on a clinical case-file data set, aligns with expert assessments. Since LAMP operates without requiring access to model gradients, logits, or internal activations, it serves as a practical and lightweight framework for auditing proprietary language models, and enabling assessment of whether a model behaves consistently with the explanations it provides.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15068v1">ModelingAgent: Bridging LLMs and Mathematical Modeling for Real-World Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 36 Pages, 26 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled substantial advances in solving mathematical problems. However, existing benchmarks often fail to reflect the complexity of real-world problems, which demand open-ended, interdisciplinary reasoning and integration of computational tools. To address this gap, we introduce ModelingBench, a novel benchmark featuring real-world-inspired, open-ended problems from math modeling competitions across diverse domains, ranging from urban traffic optimization to ecosystem resource planning. These tasks require translating natural language into formal mathematical formulations, applying appropriate tools, and producing structured, defensible reports. ModelingBench also supports multiple valid solutions, capturing the ambiguity and creativity of practical modeling. We also present ModelingAgent, a multi-agent framework that coordinates tool use, supports structured workflows, and enables iterative self-refinement to generate well-grounded, creative solutions. To evaluate outputs, we further propose ModelingJudge, an expert-in-the-loop system leveraging LLMs as domain-specialized judges assessing solutions from multiple expert perspectives. Empirical results show that ModelingAgent substantially outperforms strong baselines and often produces solutions indistinguishable from those of human experts. Together, our work provides a comprehensive framework for evaluating and advancing real-world problem-solving in open-ended, interdisciplinary modeling challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14096v2">VideoPASTA: 7K Preference Pairs That Matter for Video-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Video-language models (Video-LLMs) excel at understanding video content but struggle with spatial relationships, temporal ordering, and cross-frame continuity. To address these limitations, we introduce VideoPASTA (Preference Alignment with Spatio-Temporal-Cross Frame Adversaries), a framework that enhances Video-LLMs through targeted preference optimization. VideoPASTA trains models to distinguish accurate video representations from carefully crafted adversarial examples that deliberately violate spatial, temporal, or cross-frame relationships. With only 7,020 preference pairs and Direct Preference Optimization, VideoPASTA enables models to learn robust representations that capture fine-grained spatial details and long-range temporal dynamics. Experiments demonstrate that VideoPASTA is model agnostic and significantly improves performance, for example, achieving gains of up to 3.8% on LongVideoBench, 4.1% on VideoMME, and 4.0% on MVBench, when applied to various state-of-the-art Video-LLMs. These results demonstrate that targeted alignment, rather than massive pretraining or architectural modifications, effectively addresses core video-language challenges. Notably, VideoPASTA achieves these improvements without any human annotation or captioning, relying solely on 32-frame sampling. This efficiency makes our approach a scalable plug-and-play solution that seamlessly integrates with existing models while preserving their original capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v3">Ask, Fail, Repeat: Meeseeks, an Iterative Feedback Benchmark for LLMs' Multi-turn Instruction-Following Ability</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. For complex instructions, LLMs often struggle to fulfill all requirements in a single attempt. In practice, users typically provide iterative feedback until the LLM generates a response that meets all requirements. However, existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction. To address this gap, we propose Meeseeks. Meeseeks simulates realistic human-LLM interactions through an iterative feedback framework, which enables models to self-correct based on specific requirement failures in each turn, better reflecting real-world user-end usage patterns. Meanwhile, the benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in multi-turn scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15253v2">Evaluating Judges as Evaluators: The JETTS Benchmark of LLM-as-Judges as Test-Time Scaling Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-05-21
      | 💬 ICML 2025. The first two authors contributed equally. The codebase is at https://github.com/SalesforceAIResearch/jetts-benchmark
    </div>
    <details class="paper-abstract">
      Scaling test-time computation, or affording a generator large language model (LLM) extra compute during inference, typically employs the help of external non-generative evaluators (i.e., reward models). Concurrently, LLM-judges, models trained to generate evaluations and critiques (explanations) in natural language, are becoming increasingly popular in automatic evaluation. Despite judge empirical successes, their effectiveness as evaluators in test-time scaling settings is largely unknown. In this paper, we introduce the Judge Evaluation for Test-Time Scaling (JETTS) benchmark, which evaluates judge performance in three domains (math reasoning, code generation, and instruction following) under three task settings: response reranking, step-level beam search, and critique-based response refinement. We evaluate 10 different judge models (7B-70B parameters) for 8 different base generator models (6.7B-72B parameters). Our benchmark shows that while judges are competitive with outcome reward models in reranking, they are consistently worse than process reward models in beam search procedures. Furthermore, though unique to LLM-judges, their natural language critiques are currently ineffective in guiding the generator towards better responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16067v1">How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior</a></div>
    <div class="paper-meta">
      📅 2025-05-21
    </div>
    <details class="paper-abstract">
      Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory operations that are widely used by many agent frameworks-addition, which incorporates new experiences into the memory base, and deletion, which selectively removes past experiences-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where outdated or irrelevant experiences negatively influence current tasks. Through controlled experiments, we show that combining selective addition and deletion strategies can help mitigate these negative effects, yielding an average absolute performance gain of 10% compared to naive memory growth. Furthermore, we highlight how memory management choices affect agents' behavior under challenging conditions such as task distribution shifts and constrained memory resources. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance. We also release our code to facilitate further study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14045v1">From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18062v2">LLM-hRIC: LLM-empowered Hierarchical RAN Intelligent Control for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Despite recent advances in applying large language models (LLMs) and machine learning (ML) techniques to open radio access network (O-RAN), critical challenges remain, such as insufficient cooperation between radio access network (RAN) intelligent controllers (RICs), high computational demands hindering real-time decisions, and the lack of domain-specific finetuning. Therefore, this article introduces the LLM-empowered hierarchical RIC (LLM-hRIC) framework to improve the collaboration between RICs in O-RAN. The LLM-empowered non-real-time RIC (non-RT RIC) acts as a guider, offering a strategic guidance to the near-real-time RIC (near-RT RIC) using global network information. The RL-empowered near-RT RIC acts as an implementer, combining this guidance with local real-time data to make near-RT decisions. We evaluate the feasibility and performance of the LLM-hRIC framework in an integrated access and backhaul (IAB) network setting, and finally, discuss the open challenges of the LLM-hRIC framework for O-RAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13028v2">Evaluating the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20371v2">DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory, the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT, remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompt strategies, and (3) we design precise disambiguation metrics, and study the efficacy of various prompt strategies on multiple state-of-the-art LLMs. We conduct comprehensive experiments across 4 language pairs and 13 domains, our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12981v2">From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10940v2">Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, TagCF exploits the (Multi-modal) LLM's world knowledge and logic inference ability to extract realistic tag-based virtual logic graphs that reveal dynamic and expressive knowledge of users, refining our understanding of user behaviors. On the other hand, TagCF presents empirically effective integration modules that take advantage of the extracted tag-logic information, augmenting the recommendation performance. We conduct both online experiments and offline experiments with industrial and public datasets as verification of TagCF's effectiveness, and we empirically show that the user role modeling strategy is potentially a better choice than the modeling of item topics. Additionally, we provide evidence that the extracted logic graphs are empirically a general and transferable knowledge that can benefit a wide range of recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13995v1">Social Sycophancy: A Broader Understanding of LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      A serious risk to the safety and utility of LLMs is sycophancy, i.e., excessive agreement with and flattery of the user. Yet existing work focuses on only one aspect of sycophancy: agreement with users' explicitly stated beliefs that can be compared to a ground truth. This overlooks forms of sycophancy that arise in ambiguous contexts such as advice and support-seeking, where there is no clear ground truth, yet sycophancy can reinforce harmful implicit assumptions, beliefs, or actions. To address this gap, we introduce a richer theory of social sycophancy in LLMs, characterizing sycophancy as the excessive preservation of a user's face (the positive self-image a person seeks to maintain in an interaction). We present ELEPHANT, a framework for evaluating social sycophancy across five face-preserving behaviors (emotional validation, moral endorsement, indirect language, indirect action, and accepting framing) on two datasets: open-ended questions (OEQ) and Reddit's r/AmITheAsshole (AITA). Across eight models, we show that LLMs consistently exhibit high rates of social sycophancy: on OEQ, they preserve face 47% more than humans, and on AITA, they affirm behavior deemed inappropriate by crowdsourced human judgments in 42% of cases. We further show that social sycophancy is rewarded in preference datasets and is not easily mitigated. Our work provides theoretical grounding and empirical tools (datasets and code) for understanding and addressing this under-recognized but consequential issue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05831v3">Leveraging Robust Optimization for LLM Alignment under Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13972v1">Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 in submission
    </div>
    <details class="paper-abstract">
      Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, five generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13963v1">Through a Compressed Lens: Investigating the Impact of Quantization on LLM Explainability and Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      Quantization methods are widely used to accelerate inference and streamline the deployment of large language models (LLMs). While prior research has extensively investigated the degradation of various LLM capabilities due to quantization, its effects on model explainability and interpretability, which are crucial for understanding decision-making processes, remain unexplored. To address this gap, we conduct comprehensive experiments using three common quantization techniques at distinct bit widths, in conjunction with two explainability methods, counterfactual examples and natural language explanations, as well as two interpretability approaches, knowledge memorization analysis and latent multi-hop reasoning analysis. We complement our analysis with a thorough user study, evaluating selected explainability methods. Our findings reveal that, depending on the configuration, quantization can significantly impact model explainability and interpretability. Notably, the direction of this effect is not consistent, as it strongly depends on (1) the quantization method, (2) the explainability or interpretability approach, and (3) the evaluation protocol. In some settings, human evaluation shows that quantization degrades explainability, while in others, it even leads to improvements. Our work serves as a cautionary tale, demonstrating that quantization can unpredictably affect model transparency. This insight has important implications for deploying LLMs in applications where transparency is a critical requirement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14470v2">Agent-SafetyBench: Evaluating the Safety of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as agents, their integration into interactive environments and tool use introduce new safety challenges beyond those associated with the models themselves. However, the absence of comprehensive benchmarks for evaluating agent safety presents a significant barrier to effective assessment and further improvement. In this paper, we introduce Agent-SafetyBench, a comprehensive benchmark designed to evaluate the safety of LLM agents. Agent-SafetyBench encompasses 349 interaction environments and 2,000 test cases, evaluating 8 categories of safety risks and covering 10 common failure modes frequently encountered in unsafe interactions. Our evaluation of 16 popular LLM agents reveals a concerning result: none of the agents achieves a safety score above 60%. This highlights significant safety challenges in LLM agents and underscores the considerable need for improvement. Through failure mode and helpfulness analysis, we summarize two fundamental safety defects in current LLM agents: lack of robustness and lack of risk awareness. Furthermore, our findings suggest that reliance on defense prompts alone may be insufficient to address these safety issues, emphasizing the need for more advanced and robust strategies. To drive progress in this area, Agent-SafetyBench has been released at https://github.com/thu-coai/Agent-SafetyBench/ to facilitate further research in agent safety evaluation and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13940v1">DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 22 pages, 10 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In the field of AI4Science, large-scale language models (LLMs) show great potential to parse complex scientific semantics, integrate cross-disciplinary knowledge, and assist critical task research. However, in the field of drug discovery, despite the optimization through professional data pre-training, context window expansion, and internet search, the existing LLMs are still facing challenges such as massive multi-modal and heterogeneous data processing, domain knowledge dynamic updating delay, and insufficient confidence in predicting the results of complex computational tasks. To address these challenges, we propose the DrugPilot, an LLM-based agent with parameterized reasoning for drug discovery. DrugPilot addresses key limitations of traditional end-to-end LLM prediction approaches through its parametric inference architecture. This agent system supports major phases of the drug discovery pipeline, facilitating automated planning and execution of multi-stage research tasks. To address the critical challenge of multi-modal drug data analysis (incorporating both public datasets and user-submitted data), we developed an interactive parameterized memory pool. This innovative component standardizes real-world drug data into parametric representations, simultaneously enabling efficient knowledge retrieval in multi-turn dialogue while mitigating the information loss inherent in text-based data transmission. Additionally, we created a drug instruct dataset across 8 essential drug discovery tasks for model fine-tuning and evaluation. Based on the Berkeley function calling evaluation framework, DrugPilot demonstrated the most advanced tool calling capabilities on our drug discovery tool instruction dataset, outperforming existing agents (e.g., ReAct, LoT). Specifically, it achieves task completion rates of 98.0%, 93.5%, and 64.0% on simple, multiple, and multi-turn tasks, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04174v2">On-Device LLM for Context-Aware Wi-Fi Roaming</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Roaming in Wireless LAN (Wi-Fi) is a critical yet challenging task for maintaining seamless connectivity in dynamic mobile environments. Conventional threshold-based or heuristic schemes often fail, leading to either sticky or excessive handovers. We introduce the first cross-layer use of an on-device large language model (LLM): high-level reasoning in the application layer that issues real-time actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i) context-aware AP selection, where structured prompts fuse environmental cues (e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold adjustment, where the model adaptively decides when to roam. To satisfy the tight latency and resource budgets of edge hardware, we apply a suite of optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and quantization. Experiments on indoor and outdoor datasets show that our approach surpasses legacy heuristics and DRL baselines, achieving a strong balance between roaming stability and signal quality. These findings underscore the promise of application-layer LLM reasoning for lower-layer wireless control in future edge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13921v1">APEX: Empowering LLMs with Physics-Based Task Planning for Real-time Insight</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong reasoning and task planning capabilities but remain fundamentally limited in physical interaction modeling. Existing approaches integrate perception via Vision-Language Models (VLMs) or adaptive decision-making through Reinforcement Learning (RL), but they fail to capture dynamic object interactions or require task-specific training, limiting their real-world applicability. We introduce APEX (Anticipatory Physics-Enhanced Execution), a framework that equips LLMs with physics-driven foresight for real-time task planning. APEX constructs structured graphs to identify and model the most relevant dynamic interactions in the environment, providing LLMs with explicit physical state updates. Simultaneously, APEX provides low-latency forward simulations of physically feasible actions, allowing LLMs to select optimal strategies based on predictive outcomes rather than static observations. We evaluate APEX on three benchmarks designed to assess perception, prediction, and decision-making: (1) Physics Reasoning Benchmark, testing causal inference and object motion prediction; (2) Tetris, evaluating whether physics-informed prediction enhances decision-making performance in long-horizon planning tasks; (3) Dynamic Obstacle Avoidance, assessing the immediate integration of perception and action feasibility analysis. APEX significantly outperforms standard LLMs and VLM-based models, demonstrating the necessity of explicit physics reasoning for bridging the gap between language-based intelligence and real-world task execution. The source code and experiment setup are publicly available at https://github.com/hwj20/APEX_EXP .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13442v2">TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted to ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 64% and 44% in their respective worst-case scenarios under zero-shot setting. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems. The dataset generation code and sample data are available at https://github.com/j-bagel/treecut-math.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13890v1">Mapping the Minds of LLMs: A Graph-Based Analysis of Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Recent advances in test-time scaling have enabled Large Language Models (LLMs) to display sophisticated reasoning abilities via extended Chain-of-Thought (CoT) generation. Despite their potential, these Reasoning LLMs (RLMs) often demonstrate counterintuitive and unstable behaviors, such as performance degradation under few-shot prompting, that challenge our current understanding of RLMs. In this work, we introduce a unified graph-based analytical framework for better modeling the reasoning processes of RLMs. Our method first clusters long, verbose CoT outputs into semantically coherent reasoning steps, then constructs directed reasoning graphs to capture contextual and logical dependencies among these steps. Through comprehensive analysis across models and prompting regimes, we reveal that structural properties, such as exploration density, branching, and convergence ratios, strongly correlate with reasoning accuracy. Our findings demonstrate how prompting strategies substantially reshape the internal reasoning structure of RLMs, directly affecting task outcomes. The proposed framework not only enables quantitative evaluation of reasoning quality beyond conventional metrics but also provides practical insights for prompt engineering and the cognitive analysis of LLMs. Code and resources will be released to facilitate future research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00054v2">Automating Intervention Discovery from Scientific Literature: A Progressive Ontology Prompting and Dual-LLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted by IJCAI2025
    </div>
    <details class="paper-abstract">
      Identifying effective interventions from the scientific literature is challenging due to the high volume of publications, specialized terminology, and inconsistent reporting formats, making manual curation laborious and prone to oversight. To address this challenge, this paper proposes a novel framework leveraging large language models (LLMs), which integrates a progressive ontology prompting (POP) algorithm with a dual-agent system, named LLM-Duo. On the one hand, the POP algorithm conducts a prioritized breadth-first search (BFS) across a predefined ontology, generating structured prompt templates and action sequences to guide the automatic annotation process. On the other hand, the LLM-Duo system features two specialized LLM agents, an explorer and an evaluator, working collaboratively and adversarially to continuously refine annotation quality. We showcase the real-world applicability of our framework through a case study focused on speech-language intervention discovery. Experimental results show that our approach surpasses advanced baselines, achieving more accurate and comprehensive annotations through a fully automated process. Our approach successfully identified 2,421 interventions from a corpus of 64,177 research articles in the speech-language pathology domain, culminating in the creation of a publicly accessible intervention knowledge base with great potential to benefit the speech-language pathology community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13866v1">Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining KV cache that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60$\times$ compared to the inference with full KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at https://github.com/jiwonsong-dev/ReasoningPathCompression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13862v1">PandaGuard: Systematic Evaluation of LLM Safety in the Era of Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12043v2">MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. Continual Pre-Training (CPT) approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain-corpus to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14733v1">The Energy Cost of Reasoning: Analyzing Energy Usage in LLMs with Test-time Compute</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Scaling large language models (LLMs) has driven significant advancements, yet it faces diminishing returns and escalating energy demands. This work introduces test-time compute (TTC)-allocating additional computational resources during inference-as a compelling complement to conventional scaling strategies. Specifically, we investigate whether employing TTC can achieve superior accuracy-energy trade-offs compared to simply increasing model size. Our empirical analysis reveals that TTC surpasses traditional model scaling in accuracy/energy efficiency, with notable gains in tasks demanding complex reasoning rather than mere factual recall. Further, we identify a critical interaction between TTC performance and output sequence length, demonstrating that strategically adjusting compute resources at inference time according to query complexity can substantially enhance efficiency. Our findings advocate for TTC as a promising direction, enabling more sustainable, accurate, and adaptable deployment of future language models without incurring additional pretraining costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17662v5">PLAYER*: Enhancing LLM-based Multi-Agent Communication and Interaction in Murder Mystery Games</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      We introduce WellPlay, a reasoning dataset for multi-agent conversational inference in Murder Mystery Games (MMGs). WellPlay comprises 1,482 inferential questions across 12 games, spanning objectives, reasoning, and relationship understanding, and establishes a systematic benchmark for evaluating agent reasoning abilities in complex social settings. Building on this foundation, we present PLAYER*, a novel framework for Large Language Model (LLM)-based agents in MMGs. MMGs pose unique challenges, including undefined state spaces, absent intermediate rewards, and the need for strategic reasoning through natural language. PLAYER* addresses these challenges with a sensor-based state representation and an information-driven strategy that optimises questioning and suspect pruning. Experiments show that PLAYER* outperforms existing methods in reasoning accuracy, efficiency, and agent-human interaction, advancing reasoning agents for complex social scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12331v2">OSS-Bench: Benchmark Generator for Coding LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      In light of the rapid adoption of AI coding assistants, LLM-assisted development has become increasingly prevalent, creating an urgent need for robust evaluation of generated code quality. Existing benchmarks often require extensive manual effort to create static datasets, rely on indirect or insufficiently challenging tasks, depend on non-scalable ground truth, or neglect critical low-level security evaluations, particularly memory-safety issues. In this work, we introduce OSS-Bench, a benchmark generator that automatically constructs large-scale, live evaluation tasks from real-world open-source software. OSS-Bench replaces functions with LLM-generated code and evaluates them using three natural metrics: compilability, functional correctness, and memory safety, leveraging robust signals like compilation failures, test-suite violations, and sanitizer alerts as ground truth. In our evaluation, the benchmark, instantiated as OSS-Bench(php) and OSS-Bench(sql), profiles 17 diverse LLMs, revealing insights such as intra-family behavioral patterns and inconsistencies between model size and performance. Our results demonstrate that OSS-Bench mitigates overfitting by leveraging the evolving complexity of OSS and highlights LLMs' limited understanding of low-level code security via extended fuzzing experiments. Overall, OSS-Bench offers a practical and scalable framework for benchmarking the real-world coding capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13830v1">Improving Noise Robustness of LLM-based Zero-shot TTS via Discrete Acoustic Token Denoising</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted by Interspeech 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based zero-shot text-to-speech (TTS) methods tend to preserve the acoustic environment of the audio prompt, leading to degradation in synthesized speech quality when the audio prompt contains noise. In this paper, we propose a novel neural codec-based speech denoiser and integrate it with the advanced LLM-based TTS model, LauraTTS, to achieve noise-robust zero-shot TTS. The proposed codec denoiser consists of an audio codec, a token denoiser, and an embedding refiner. The token denoiser predicts the first two groups of clean acoustic tokens from the noisy ones, which can serve as the acoustic prompt for LauraTTS to synthesize high-quality personalized speech or be converted to clean speech waveforms through the embedding refiner and codec decoder. Experimental results show that our proposed codec denoiser outperforms state-of-the-art speech enhancement (SE) methods, and the proposed noise-robust LauraTTS surpasses the approach using additional SE models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13819v1">Fragments to Facts: Partial-Information Fragment Inference from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can leak sensitive training data through memorization and membership inference attacks. Prior work has primarily focused on strong adversarial assumptions, including attacker access to entire samples or long, ordered prefixes, leaving open the question of how vulnerable LLMs are when adversaries have only partial, unordered sample information. For example, if an attacker knows a patient has "hypertension," under what conditions can they query a model fine-tuned on patient data to learn the patient also has "osteoarthritis?" In this paper, we introduce a more general threat model under this weaker assumption and show that fine-tuned LLMs are susceptible to these fragment-specific extraction attacks. To systematically investigate these attacks, we propose two data-blind methods: (1) a likelihood ratio attack inspired by methods from membership inference, and (2) a novel approach, PRISM, which regularizes the ratio by leveraging an external prior. Using examples from both medical and legal settings, we show that both methods are competitive with a data-aware baseline classifier that assumes access to labeled in-distribution data, underscoring their robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14476v2">DAPO: An Open-Source LLM Reinforcement Learning System at Scale</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Project Page: https://dapo-sia.github.io/
    </div>
    <details class="paper-abstract">
      Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13794v1">LLM-based Evaluation Policy Extraction for Ecological Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Evaluating ecological time series is critical for benchmarking model performance in many important applications, including predicting greenhouse gas fluxes, capturing carbon-nitrogen dynamics, and monitoring hydrological cycles. Traditional numerical metrics (e.g., R-squared, root mean square error) have been widely used to quantify the similarity between modeled and observed ecosystem variables, but they often fail to capture domain-specific temporal patterns critical to ecological processes. As a result, these methods are often accompanied by expert visual inspection, which requires substantial human labor and limits the applicability to large-scale evaluation. To address these challenges, we propose a novel framework that integrates metric learning with large language model (LLM)-based natural language policy extraction to develop interpretable evaluation criteria. The proposed method processes pairwise annotations and implements a policy optimization mechanism to generate and combine different assessment metrics. The results obtained on multiple datasets for evaluating the predictions of crop gross primary production and carbon dioxide flux have confirmed the effectiveness of the proposed method in capturing target assessment preferences, including both synthetically generated and expert-annotated model comparisons. The proposed framework bridges the gap between numerical metrics and expert knowledge while providing interpretable evaluation policies that accommodate the diverse needs of different ecosystem modeling studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14727v1">The Evolution of Alpha in Finance Harnessing Human Insight and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      The pursuit of alpha returns that exceed market benchmarks has undergone a profound transformation, evolving from intuition-driven investing to autonomous, AI powered systems. This paper introduces a comprehensive five stage taxonomy that traces this progression across manual strategies, statistical models, classical machine learning, deep learning, and agentic architectures powered by large language models (LLMs). Unlike prior surveys focused narrowly on modeling techniques, this review adopts a system level lens, integrating advances in representation learning, multimodal data fusion, and tool augmented LLM agents. The strategic shift from static predictors to contextaware financial agents capable of real time reasoning, scenario simulation, and cross modal decision making is emphasized. Key challenges in interpretability, data fragility, governance, and regulatory compliance areas critical to production deployment are examined. The proposed taxonomy offers a unified framework for evaluating maturity, aligning infrastructure, and guiding the responsible development of next generation alpha systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v1">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14925v1">Too Long, Didn't Model: Decomposing LLM Long-Context Understanding With Novels</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Although the context length of large language models (LLMs) has increased to millions of tokens, evaluating their effectiveness beyond needle-in-a-haystack approaches has proven difficult. We argue that novels provide a case study of subtle, complicated structure and long-range semantic dependencies often over 128k tokens in length. Inspired by work on computational novel analysis, we release the Too Long, Didn't Model (TLDM) benchmark, which tests a model's ability to report plot summary, storyworld configuration, and elapsed narrative time. We find that none of seven tested frontier LLMs retain stable understanding beyond 64k tokens. Our results suggest language model developers must look beyond "lost in the middle" benchmarks when evaluating model performance in complex long-context scenarios. To aid in further development we release the TLDM benchmark together with reference code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14918v1">Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14899v1">Think, Reflect, Create: Metacognitive Learning for Zero-Shot Robotic Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown great potential across various domains, their applications in robotics remain largely limited to static, prompt-based behaviors and still face challenges in handling complex tasks under zero-shot or few-shot settings. Inspired by human metacognitive learning and creative problem-solving, we address this limitation by exploring a fundamental research question: Can LLMs be empowered with metacognitive capabilities to reason, reflect, and create, thereby enhancing their ability to perform robotic tasks with minimal demonstrations? In this paper, we present an early-stage framework that integrates metacognitive learning into LLM-powered multi-robot collaboration. The proposed framework equips the LLM-powered robotic agents with a skill decomposition and self-reflection mechanism that identifies modular skills from prior tasks, reflects on failures in unseen task scenarios, and synthesizes effective new solutions. Experimental results show that our metacognitive-learning-empowered LLM framework significantly outperforms existing baselines. Moreover, we observe that the framework is capable of generating solutions that differ from the ground truth yet still successfully complete the tasks. These exciting findings support our hypothesis that metacognitive learning can foster creativity in robotic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19721v2">Sensing and Steering Stereotypes: Extracting and Applying Gender Representation Vectors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to perpetuate stereotypes and exhibit biases. Various strategies have been proposed to mitigate these biases, but most work studies biases in LLMs as a black-box problem without considering how concepts are represented within the model. We adapt techniques from representation engineering to study how the concept of "gender" is represented within LLMs. We introduce a new method that extracts concept representations via probability weighting without labeled data and efficiently selects a steering vector for measuring and manipulating the model's representation. We also present a projection-based method that enables precise steering of model predictions and demonstrate its effectiveness in mitigating gender bias in LLMs. Our code is available at: https://github.com/hannahxchen/gender-bias-steering
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14884v1">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12264v2">Uncertainty quantification in fine-tuned LLMs using LoRA ensembles</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 Accepted for ICLR2025 Workshop "Quantify Uncertainty and Hallucination in Foundation Models: The Next Frontier in Reliable AI"
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and balance between retained prior knowledge and domain specific adaptation during and after fine-tuning. We identify unexpected retention of acquired knowledge during fine-tuning in the overfitting regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14864v1">Balanced and Elastic End-to-end Training of Dynamic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
    </div>
    <details class="paper-abstract">
      To reduce computational and memory costs in Large Language Models (LLMs), dynamic workload reduction schemes like Mixture of Experts (MoEs), parameter pruning, layer freezing, sparse attention, early token exit, and Mixture of Depths (MoDs) have emerged. However, these methods introduce severe workload imbalances, limiting their practicality for large-scale distributed training. We propose DynMo, an autonomous dynamic load balancing solution that ensures optimal compute distribution when using pipeline parallelism in training dynamic models. DynMo adaptively balances workloads, dynamically packs tasks into fewer workers to free idle resources, and supports both multi-GPU single-node and multi-node systems. Compared to static training methods (Megatron-LM, DeepSpeed), DynMo accelerates training by up to 1.23x (MoEs), 3.18x (pruning), 2.23x (layer freezing), 4.02x (sparse attention), 4.52x (early exit), and 1.17x (MoDs). DynMo is available at https://anonymous.4open.science/r/DynMo-4D04/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14832v1">SEPS: A Separability Measure for Robust Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-20
      | 💬 32 pages
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to selectively remove targeted knowledge from Large Language Models (LLMs), ensuring they forget specified content while retaining essential information. Existing unlearning metrics assess whether a model correctly answers retain queries and rejects forget queries, but they fail to capture real-world scenarios where forget queries rarely appear in isolation. In fact, forget and retain queries often coexist within the same prompt, making mixed-query evaluation crucial. We introduce SEPS, an evaluation framework that explicitly measures a model's ability to both forget and retain information within a single prompt. Through extensive experiments across three benchmarks, we identify two key failure modes in existing unlearning methods: (1) untargeted unlearning indiscriminately erases both forget and retain content once a forget query appears, and (2) targeted unlearning overfits to single-query scenarios, leading to catastrophic failures when handling multiple queries. To address these issues, we propose Mixed Prompt (MP) unlearning, a strategy that integrates both forget and retain queries into a unified training objective. Our approach significantly improves unlearning effectiveness, demonstrating robustness even in complex settings with up to eight mixed forget and retain queries in a single prompt.
    </details>
</div>
