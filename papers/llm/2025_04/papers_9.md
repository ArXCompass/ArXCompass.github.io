# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05491v1">REEF: Relevance-Aware and Efficient LLM Adapter for Video Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted at CVPRW'25
    </div>
    <details class="paper-abstract">
      Integrating vision models into large language models (LLMs) has sparked significant interest in creating vision-language foundation models, especially for video understanding. Recent methods often utilize memory banks to handle untrimmed videos for video-level understanding. However, they typically compress visual memory using similarity-based greedy approaches, which can overlook the contextual importance of individual tokens. To address this, we introduce an efficient LLM adapter designed for video-level understanding of untrimmed videos that prioritizes the contextual relevance of spatio-temporal tokens. Our framework leverages scorer networks to selectively compress the visual memory bank and filter spatial tokens based on relevance, using a differentiable Top-K operator for end-to-end training. Across three key video-level understanding tasks$\unicode{x2013}$ untrimmed video classification, video question answering, and video captioning$\unicode{x2013}$our method achieves competitive or superior results on four large-scale datasets while reducing computational overhead by up to 34%. The code will be available soon on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11007v2">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, together with their large model size, make their deployment more challenging. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design TMO, a local-cloud LLM inference system with Three-M Offloading: Multi-modal, Multi-task, and Multi-dialogue. TMO incorporates (i) a lightweight local LLM that can process simple tasks at high speed and (ii) a large-scale cloud LLM that can handle multi-modal data sources. We develop a resource-constrained reinforcement learning (RCRL) strategy for TMO that optimizes the inference location (i.e., local vs. cloud) and multi-modal data sources to use for each task/dialogue, aiming to maximize the long-term reward (response quality, latency, and usage cost) while adhering to resource constraints. We also contribute M4A1, a new dataset we curated that contains reward and cost metrics across multiple modality, task, dialogue, and LLM configurations, enabling evaluation of offloading decisions. We demonstrate the effectiveness of TMO compared to several exploration-decision and LLM-as-Agent baselines, showing significant improvements in latency, cost, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05370v1">EduPlanner: LLM-Based Multi-Agent Systems for Customized and Intelligent Instructional Design</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced smart education in the Artificial General Intelligence (AGI) era. A promising application lies in the automatic generalization of instructional design for curriculum and learning activities, focusing on two key aspects: (1) Customized Generation: generating niche-targeted teaching content based on students' varying learning abilities and states, and (2) Intelligent Optimization: iteratively optimizing content based on feedback from learning effectiveness or test scores. Currently, a single large LLM cannot effectively manage the entire process, posing a challenge for designing intelligent teaching plans. To address these issues, we developed EduPlanner, an LLM-based multi-agent system comprising an evaluator agent, an optimizer agent, and a question analyst, working in adversarial collaboration to generate customized and intelligent instructional design for curriculum and learning activities. Taking mathematics lessons as our example, EduPlanner employs a novel Skill-Tree structure to accurately model the background mathematics knowledge of student groups, personalizing instructional design for curriculum and learning activities according to students' knowledge levels and learning abilities. Additionally, we introduce the CIDDP, an LLM-based five-dimensional evaluation module encompassing clarity, Integrity, Depth, Practicality, and Pertinence, to comprehensively assess mathematics lesson plan quality and bootstrap intelligent optimization. Experiments conducted on the GSM8K and Algebra datasets demonstrate that EduPlanner excels in evaluating and optimizing instructional design for curriculum and learning activities. Ablation studies further validate the significance and effectiveness of each component within the framework. Our code is publicly available at https://github.com/Zc0812/Edu_Planner
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05352v1">Achieving binary weight and activation for LLMs using Post-Training Quantization</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Quantizing large language models (LLMs) to 1-bit precision significantly reduces computational costs, but existing quantization techniques suffer from noticeable performance degradation when using weight and activation precisions below 4 bits (W4A4). In this paper, we propose a post-training quantization framework with W(1+1)A(1*4) configuration, where weights are quantized to 1 bit with an additional 1 bit for fine-grain grouping and activations are quantized to 1 bit with a 4-fold increase in the number of channels. For weight quantization, we propose utilizing Hessian-aware fine-grained grouping along with an EM-based quantization scheme. For activation quantization, we decompose INT4-quantized activations into a 4 * INT1 format equivalently and simultaneously smooth the scaling factors based on quantization errors, which further reduces the quantization errors in activations. Our method surpasses state-of-the-art (SOTA) LLM quantization baselines on W2A4 across multiple tasks, pushing the boundaries of existing LLM quantization methods toward fully binarized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07137v1">Large Language Model (LLM) for Software Security: Code Analysis, Malware Analysis, Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as powerful tools in cybersecurity, offering advanced capabilities in malware detection, generation, and real-time monitoring. Numerous studies have explored their application in cybersecurity, demonstrating their effectiveness in identifying novel malware variants, analyzing malicious code structures, and enhancing automated threat analysis. Several transformer-based architectures and LLM-driven models have been proposed to improve malware analysis, leveraging semantic and structural insights to recognize malicious intent more accurately. This study presents a comprehensive review of LLM-based approaches in malware code analysis, summarizing recent advancements, trends, and methodologies. We examine notable scholarly works to map the research landscape, identify key challenges, and highlight emerging innovations in LLM-driven cybersecurity. Additionally, we emphasize the role of static analysis in malware detection, introduce notable datasets and specialized LLM models, and discuss essential datasets supporting automated malware research. This study serves as a valuable resource for researchers and cybersecurity professionals, offering insights into LLM-powered malware detection and defence strategies while outlining future directions for strengthening cybersecurity resilience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07135v1">SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05276v1">Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Short answer assessment is a vital component of science education, allowing evaluation of students' complex three-dimensional understanding. Large language models (LLMs) that possess human-like ability in linguistic tasks are increasingly popular in assisting human graders to reduce their workload. However, LLMs' limitations in domain knowledge restrict their understanding in task-specific requirements and hinder their ability to achieve satisfactory performance. Retrieval-augmented generation (RAG) emerges as a promising solution by enabling LLMs to access relevant domain-specific knowledge during assessment. In this work, we propose an adaptive RAG framework for automated grading that dynamically retrieves and incorporates domain-specific knowledge based on the question and student answer context. Our approach combines semantic search and curated educational sources to retrieve valuable reference materials. Experimental results in a science education dataset demonstrate that our system achieves an improvement in grading accuracy compared to baseline LLM approaches. The findings suggest that RAG-enhanced grading systems can serve as reliable support with efficient performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05262v1">Do PhD-level LLMs Truly Grasp Elementary Addition? Probing Rule Learning vs. Memorization in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Despite high benchmark scores, Large Language Models (LLMs) often fail simple problem, raising a critical question: Do LLMs learn mathematical principles or merely memorize patterns? Rather than designing increasingly complex benchmarks like recent works, we investigate this using elementary two-integer addition ($0$ to $2^{64}$), probing two core properties: commutativity ($A+B=B+A$) and compositional generalization (via isomorphic symbolic mappings, e.g., $7 \rightarrow y$). While state-of-the-art LLMs achieve 73.8-99.8\% accuracy on numerical addition, performance collapses to $\leq$7.5\% under symbolic mapping, indicating failure to generalize learned rules. Non-monotonic performance scaling with digit count and frequent commutativity violations (over 1,700 cases of $A+B \neq B+A$) further support this. Explicitly providing addition rules degrades performance by 81.2\% on average, while self-explanation maintains baseline accuracy, suggesting LLM arithmetic processing is misaligned with human-defined principles. Our findings indicate current LLMs rely on memory pattern over genuine rule learning, highlighting architectural limitations and the need for new approaches to achieve true mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05259v1">How to evaluate control measures for LLM agents? A trajectory from today to superintelligence</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      As LLM agents grow more capable of causing harm autonomously, AI developers will rely on increasingly sophisticated control measures to prevent possibly misaligned agents from causing harm. AI developers could demonstrate that their control measures are sufficient by running control evaluations: testing exercises in which a red team produces agents that try to subvert control measures. To ensure control evaluations accurately capture misalignment risks, the affordances granted to this red team should be adapted to the capability profiles of the agents to be deployed under control measures. In this paper we propose a systematic framework for adapting affordances of red teams to advancing AI capabilities. Rather than assuming that agents will always execute the best attack strategies known to humans, we demonstrate how knowledge of an agents's actual capability profile can inform proportional control evaluations, resulting in more practical and cost-effective control measures. We illustrate our framework by considering a sequence of five fictional models (M1-M5) with progressively advanced capabilities, defining five distinct AI control levels (ACLs). For each ACL, we provide example rules for control evaluation, control measures, and safety cases that could be appropriate. Finally, we show why constructing a compelling AI control safety case for superintelligent LLM agents will require research breakthroughs, highlighting that we might eventually need alternative approaches to mitigating misalignment risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05239v1">LLM-based Automated Grading with Human-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09439v2">Spider: Any-to-Many Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) have emerged as an extension of Large Language Models (LLMs), enabling the integration of various modalities. However, Any-to-Any MLLMs are limited to generating pairwise modalities 'Text + X' within a single response, such as Text + {Image or Audio or Video}. To address this limitation, we introduce Spider, a novel efficient Any-to-Many Modalities Generation (AMMG) framework, which can generate an arbitrary combination of modalities 'Text + Xs', such as Text + {Image and Audio and Video}. To achieve efficient AMMG, our Spider integrates three core components: a Base Model for basic X-to-X (i.e., Any-to-Any) modality processing, an Any-to-Many Instruction Template designed for producing Xs signal prompts, and a novel Efficient Decoders-Controller for controlling multimodal Decoders to generate Xs (many-modal) contents. To train Spider, we constructed a novel Text-formatted Many-Modal (TMM) dataset, which facilitates learning the X-to-Xs (i.e., Any-to-Many) capability necessary for AMMG. Ultimately, the well-trained Spider generates a pseudo X-to-Xs dataset, the first-ever X-to-Xs many-modal dataset, enhancing the potential for AMMG tasks in future research. Overall, this work not only pushes the boundary of multimodal interaction but also provides rich data support for advancing the field. Code: https://github.com/Layjins/Spider
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v1">Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05216v1">Unleashing the Power of LLMs in Dense Retrieval with Query Likelihood Modeling</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Dense retrieval is a crucial task in Information Retrieval (IR) and is the foundation for downstream tasks such as re-ranking. Recently, large language models (LLMs) have shown compelling semantic understanding capabilities and are appealing to researchers studying dense retrieval. LLMs, as decoder-style generative models, are competent at language generation while falling short on modeling global information due to the lack of attention to tokens afterward. Inspired by the classical word-based language modeling approach for IR, i.e., the query likelihood (QL) model, we seek to sufficiently utilize LLMs' generative ability by QL maximization. However, instead of ranking documents with QL estimation, we introduce an auxiliary task of QL maximization to yield a better backbone for contrastively learning a discriminative retriever. We name our model as LLM-QL. To condense global document semantics to a single vector during QL modeling, LLM-QL has two major components, Attention Stop (AS) and Input Corruption (IC). AS stops the attention of predictive tokens to previous tokens until the ending token of the document. IC masks a portion of tokens in the input documents during prediction. Experiments on MSMARCO show that LLM-QL can achieve significantly better performance than other LLM-based retrievers and using QL estimated by LLM-QL for ranking outperforms word-based QL by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07045v2">Scalable and Ethical Insider Threat Detection through Data Synthesis and Analysis by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 6 pages, 0 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Insider threats wield an outsized influence on organizations, disproportionate to their small numbers. This is due to the internal access insiders have to systems, information, and infrastructure. %One example of this influence is where anonymous respondents submit web-based job search site reviews, an insider threat risk to organizations. Signals for such risks may be found in anonymous submissions to public web-based job search site reviews. This research studies the potential for large language models (LLMs) to analyze and detect insider threat sentiment within job site reviews. Addressing ethical data collection concerns, this research utilizes synthetic data generation using LLMs alongside existing job review datasets. A comparative analysis of sentiment scores generated by LLMs is benchmarked against expert human scoring. Findings reveal that LLMs demonstrate alignment with human evaluations in most cases, thus effectively identifying nuanced indicators of threat sentiment. The performance is lower on human-generated data than synthetic data, suggesting areas for improvement in evaluating real-world data. Text diversity analysis found differences between human-generated and LLM-generated datasets, with synthetic data exhibiting somewhat lower diversity. Overall, the results demonstrate the applicability of LLMs to insider threat detection, and a scalable solution for insider sentiment testing by overcoming ethical and logistical barriers tied to data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05204v1">Quantum Program Linting with LLMs: Emerging Results from a Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Ensuring the quality of quantum programs is increasingly important; however, traditional static analysis techniques are insufficient due to the unique characteristics of quantum computing. Quantum-specific linting tools, such as LintQ, have been developed to detect quantum-specific programming problems; however, they typically rely on manually crafted analysis queries. The manual effort required to update these tools limits their adaptability to evolving quantum programming practices. To address this challenge, this study investigates the feasibility of employing Large Language Models (LLMs) to develop a novel linting technique for quantum software development and explores potential avenues to advance linting approaches. We introduce LintQ-LLM, an LLM-based linting tool designed to detect quantum-specific problems comparable to those identified by LintQ. Through an empirical comparative study using real-world Qiskit programs, our results show that LintQ-LLM is a viable solution that complements LintQ, with particular strengths in problem localization, explanation clarity, and adaptability potential for emerging quantum programming frameworks, thus providing a basis for further research. Furthermore, this study discusses several research opportunities for developing more advanced, adaptable, and feedback-aware quantum software quality assurance methods by leveraging LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05147v1">Pr$εε$mpt: Sanitizing Sensitive Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has introduced new privacy challenges, particularly during inference where sensitive information in prompts may be exposed to proprietary LLM APIs. In this paper, we address the problem of formally protecting the sensitive information contained in a prompt while maintaining response quality. To this end, first, we introduce a cryptographically inspired notion of a prompt sanitizer which transforms an input prompt to protect its sensitive tokens. Second, we propose Pr$\epsilon\epsilon$mpt, a novel system that implements a prompt sanitizer. Pr$\epsilon\epsilon$mpt categorizes sensitive tokens into two types: (1) those where the LLM's response depends solely on the format (such as SSNs, credit card numbers), for which we use format-preserving encryption (FPE); and (2) those where the response depends on specific values, (such as age, salary) for which we apply metric differential privacy (mDP). Our evaluation demonstrates that Pr$\epsilon\epsilon$mpt is a practical method to achieve meaningful privacy guarantees, while maintaining high utility compared to unsanitized prompts, and outperforming prior methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05193v3">RevisEval: Improving LLM-as-a-Judge via Response-Adapted References</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      With significant efforts in recent studies, LLM-as-a-Judge has become a cost-effective alternative to human evaluation for assessing text generation quality in a wide range of tasks. However, there still remains a reliability gap between LLM-as-a-Judge and human evaluation. One important reason is the lack of guided oracles in the evaluation process. Motivated by the role of reference pervasively used in classic text evaluation, we introduce RevisEval, a novel text generation evaluation paradigm via the response-adapted references. RevisEval is driven by the key observation that an ideal reference should maintain the necessary relevance to the response to be evaluated. Specifically, RevisEval leverages the text revision capabilities of large language models (LLMs) to adaptively revise the response, then treat the revised text as the reference (response-adapted reference) for the subsequent evaluation. Extensive experiments demonstrate that RevisEval outperforms traditional reference-free and reference-based evaluation paradigms that use LLM-as-a-Judge across NLG tasks and open-ended instruction-following tasks. More importantly, our response-adapted references can further boost the classical text metrics, e.g., BLEU and BERTScore, compared to traditional references and even rival the LLM-as-a-Judge. A detailed analysis is also conducted to confirm RevisEval's effectiveness in bias reduction, the impact of inference cost, and reference relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05108v1">Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Discovering efficient algorithms for solving complex problems has been an outstanding challenge in mathematics and computer science, requiring substantial human expertise over the years. Recent advancements in evolutionary search with large language models (LLMs) have shown promise in accelerating the discovery of algorithms across various domains, particularly in mathematics and optimization. However, existing approaches treat the LLM as a static generator, missing the opportunity to update the model with the signal obtained from evolutionary exploration. In this work, we propose to augment LLM-based evolutionary search by continuously refining the search operator - the LLM - through reinforcement learning (RL) fine-tuning. Our method leverages evolutionary search as an exploration strategy to discover improved algorithms, while RL optimizes the LLM policy based on these discoveries. Our experiments on three combinatorial optimization tasks - bin packing, traveling salesman, and the flatpack problem - show that combining RL and evolutionary search improves discovery efficiency of improved algorithms, showcasing the potential of RL-enhanced evolutionary strategies to assist computer scientists and mathematicians for more efficient algorithm design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05047v1">Debate Only When Necessary: Adaptive Multiagent Collaboration for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Multiagent collaboration has emerged as a promising framework for enhancing the reasoning capabilities of large language models (LLMs). While this approach improves reasoning capability, it incurs substantial computational overhead due to iterative agent interactions. Furthermore, engaging in debates for queries that do not necessitate collaboration amplifies the risk of error generation. To address these challenges, we propose Debate Only When Necessary (DOWN), an adaptive multiagent debate framework that selectively activates the debate process based on the confidence score of the agent's initial response. For queries where debate is triggered, agents refine their outputs using responses from participating agents and their confidence scores. Experimental results demonstrate that this mechanism significantly improves efficiency while maintaining or even surpassing the performance of existing multiagent debate systems. We also find that confidence-guided debate mitigates error propagation and enhances the selective incorporation of reliable responses. These results establish DOWN as an optimization strategy for efficient and effective multiagent reasoning, facilitating the practical deployment of LLM-based collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05006v1">Enhancing Smart Contract Vulnerability Detection in DApps Leveraging Fine-Tuned LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Decentralized applications (DApps) face significant security risks due to vulnerabilities in smart contracts, with traditional detection methods struggling to address emerging and machine-unauditable flaws. This paper proposes a novel approach leveraging fine-tuned Large Language Models (LLMs) to enhance smart contract vulnerability detection. We introduce a comprehensive dataset of 215 real-world DApp projects (4,998 contracts), including hard-to-detect logical errors like token price manipulation, addressing the limitations of existing simplified benchmarks. By fine-tuning LLMs (Llama3-8B and Qwen2-7B) with Full-Parameter Fine-Tuning (FFT) and Low-Rank Adaptation (LoRA), our method achieves superior performance, attaining an F1-score of 0.83 with FFT and data augmentation via Random Over Sampling (ROS). Comparative experiments demonstrate significant improvements over prompt-based LLMs and state-of-the-art tools. Notably, the approach excels in detecting non-machine-auditable vulnerabilities, achieving 0.97 precision and 0.68 recall for price manipulation flaws. The results underscore the effectiveness of domain-specific LLM fine-tuning and data augmentation in addressing real-world DApp security challenges, offering a robust solution for blockchain ecosystem protection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04994v1">Following the Whispers of Values: Unraveling Neural Mechanisms Behind Value-Oriented Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of large language models (LLMs), they can present unintended biases and harmful behaviors driven by encoded values, emphasizing the urgent need to understand the value mechanisms behind them. However, current research primarily evaluates these values through external responses with a focus on AI safety, lacking interpretability and failing to assess social values in real-world contexts. In this paper, we propose a novel framework called ValueExploration, which aims to explore the behavior-driven mechanisms of National Social Values within LLMs at the neuron level. As a case study, we focus on Chinese Social Values and first construct C-voice, a large-scale bilingual benchmark for identifying and evaluating Chinese Social Values in LLMs. By leveraging C-voice, we then identify and locate the neurons responsible for encoding these values according to activation difference. Finally, by deactivating these neurons, we analyze shifts in model behavior, uncovering the internal mechanism by which values influence LLM decision-making. Extensive experiments on four representative LLMs validate the efficacy of our framework. The benchmark and code will be available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02205v3">DataLab: A Unified Platform for LLM-Powered Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted to ICDE 2025
    </div>
    <details class="paper-abstract">
      Business intelligence (BI) transforms large volumes of data within modern organizations into actionable insights for informed decision-making. Recently, large language model (LLM)-based agents have streamlined the BI workflow by automatically performing task planning, reasoning, and actions in executable environments based on natural language (NL) queries. However, existing approaches primarily focus on individual BI tasks such as NL2SQL and NL2VIS. The fragmentation of tasks across different data roles and tools lead to inefficiencies and potential errors due to the iterative and collaborative nature of BI. In this paper, we introduce DataLab, a unified BI platform that integrates a one-stop LLM-based agent framework with an augmented computational notebook interface. DataLab supports various BI tasks for different data roles in data preparation, analysis, and visualization by seamlessly combining LLM assistance with user customization within a single environment. To achieve this unification, we design a domain knowledge incorporation module tailored for enterprise-specific BI tasks, an inter-agent communication mechanism to facilitate information sharing across the BI workflow, and a cell-based context management strategy to enhance context utilization efficiency in BI notebooks. Extensive experiments demonstrate that DataLab achieves state-of-the-art performance on various BI tasks across popular research benchmarks. Moreover, DataLab maintains high effectiveness and efficiency on real-world datasets from Tencent, achieving up to a 58.58% increase in accuracy and a 61.65% reduction in token cost on enterprise-specific BI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04953v1">M-Prometheus: A Suite of Open Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on natively multilingual feedback data instead of translated data. We release our models, training dataset, and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18666v2">AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identifying 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04915v1">Collab-RAG: Boosting Retrieval-Augmented Generation for Complex Question Answering via White-Box and Black-Box LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Work in progress. Code: https://github.com/ritaranx/Collab-RAG/
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems often struggle to handle multi-hop question-answering tasks accurately due to irrelevant context retrieval and limited complex reasoning capabilities. We introduce Collab-RAG, a collaborative training framework that leverages mutual enhancement between a white-box small language model (SLM) and a blackbox large language model (LLM) for RAG. Specifically, the SLM decomposes complex queries into simpler sub-questions, thus enhancing the accuracy of the retrieval and facilitating more effective reasoning by the black-box LLM. Concurrently, the black-box LLM provides feedback signals to improve the SLM's decomposition capability. We observe that Collab-RAG relies solely on supervision from an affordable black-box LLM without additional distillation from frontier LLMs, yet demonstrates strong generalization across multiple black-box LLMs. Experimental evaluations across five multi-hop QA datasets demonstrate that Collab-RAG substantially outperforms existing black-box-only and SLM fine-tuning baselines by 1.8%-14.2% on average. In particular, our fine-tuned 3B SLM surpasses a frozen 32B LLM in question decomposition, highlighting the efficiency of Collab-RAG in improving reasoning and retrieval for complex questions. The code of Collab-RAG is available on https://github.com/ritaranx/Collab-RAG/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04877v1">SoK: LLM-based Log Parsing</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 34 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Log data, generated by software systems, provides crucial insights for tasks like monitoring, root cause analysis, and anomaly detection. Due to the vast volume of logs, automated log parsing is essential to transform semi-structured log messages into structured representations. Traditional log parsing techniques often require manual configurations, such as defining log formats or labeling data, which limits scalability and usability. Recent advances in large language models (LLMs) have introduced the new research field of LLM-based log parsing, offering potential improvements in automation and adaptability. Despite promising results, there is no structured overview of these approaches since this is a relatively new research field with the earliest advances published in late 2023. This paper systematically reviews 29 LLM-based log parsing methods, comparing their capabilities, limitations, and reliance on manual effort. We analyze the learning and prompt-engineering paradigms employed, efficiency- and effectiveness-enhancing techniques, and the role of LLMs in the parsing process. We aggregate the results of the survey in a large table comprising the characterizing features of LLM-based log parsing approaches and derive the general process of LLM-based log parsing, incorporating all reviewed approaches in a single flow chart. Additionally, we benchmark seven open-source LLM-based log parsers on public datasets and critically assess their reproducibility. Our findings summarize the advances of this new research field and provide insights for researchers and practitioners seeking efficient and user-friendly log parsing solutions, with all code and results made publicly available for transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04855v1">BIASINSPECTOR: Detecting Bias in Structured Data through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 21 pages,6 figures
    </div>
    <details class="paper-abstract">
      Detecting biases in structured data is a complex and time-consuming task. Existing automated techniques are limited in diversity of data types and heavily reliant on human case-by-case handling, resulting in a lack of generalizability. Currently, large language model (LLM)-based agents have made significant progress in data science, but their ability to detect data biases is still insufficiently explored. To address this gap, we introduce the first end-to-end, multi-agent synergy framework, BIASINSPECTOR, designed for automatic bias detection in structured data based on specific user requirements. It first develops a multi-stage plan to analyze user-specified bias detection tasks and then implements it with a diverse and well-suited set of tools. It delivers detailed results that include explanations and visualizations. To address the lack of a standardized framework for evaluating the capability of LLM agents to detect biases in data, we further propose a comprehensive benchmark that includes multiple evaluation metrics and a large set of test cases. Extensive experiments demonstrate that our framework achieves exceptional overall performance in structured data bias detection, setting a new milestone for fairer data applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04815v1">Beyond Answers: How LLMs Can Pursue Strategic Thinking in Education</a></div>
    <div class="paper-meta">
      📅 2025-04-07
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) holds transformative potential in education, enabling personalized learning, enhancing inclusivity, and encouraging creativity and curiosity. In this paper, we explore how Large Language Models (LLMs) can act as both patient tutors and collaborative partners to enhance education delivery. As tutors, LLMs personalize learning by offering step-by-step explanations and addressing individual needs, making education more inclusive for students with diverse backgrounds or abilities. As collaborators, they expand students' horizons, supporting them in tackling complex, real-world problems and co-creating innovative projects. However, to fully realize these benefits, LLMs must be leveraged not as tools for providing direct solutions but rather to guide students in developing resolving strategies and finding learning paths together. Therefore, a strong emphasis should be placed on educating students and teachers on the successful use of LLMs to ensure their effective integration into classrooms. Through practical examples and real-world case studies, this paper illustrates how LLMs can make education more inclusive and engaging while empowering students to reach their full potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17003v5">Safety Layers in Aligned Large Language Models: The Key to LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Accepted by ICLR 2025. The code is available at https://github.com/listen0425/Safety-Layers
    </div>
    <details class="paper-abstract">
      Aligned LLMs are secure, capable of recognizing and refusing to answer malicious questions. However, the role of internal parameters in maintaining such security is not well understood yet, further these models can be vulnerable to security degradation when subjected to fine-tuning attacks. To address these challenges, our work uncovers the mechanism behind security in aligned LLMs at the parameter level, identifying a small set of contiguous layers in the middle of the model that are crucial for distinguishing malicious queries from normal ones, referred to as ``safety layers". We first confirm the existence of these safety layers by analyzing variations in input vectors within the model's internal layers. Additionally, we leverage the over-rejection phenomenon and parameters scaling analysis to precisely locate the safety layers. Building on these findings, we propose a novel fine-tuning approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient of the safety layers during fine-tuning to address the security degradation. Our experiments demonstrate that the proposed approach can significantly preserve LLM security while maintaining performance and reducing computational resources compared to full fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16559v2">Demystifying Issues, Causes and Solutions in LLM Open-Source Projects</a></div>
    <div class="paper-meta">
      📅 2025-04-07
      | 💬 Preprint accepted for publication in Journal of Systems and Software, 2025
    </div>
    <details class="paper-abstract">
      With the advancements of Large Language Models (LLMs), an increasing number of open-source software projects are using LLMs as their core functional component. Although research and practice on LLMs are capturing considerable interest, no dedicated studies explored the challenges faced by practitioners of LLM open-source projects, the causes of these challenges, and potential solutions. To fill this research gap, we conducted an empirical study to understand the issues that practitioners encounter when developing and using LLM open-source software, the possible causes of these issues, and potential solutions. We collected all closed issues from 15 LLM open-source projects and labelled issues that met our requirements. We then randomly selected 994 issues from the labelled issues as the sample for data extraction and analysis to understand the prevalent issues, their underlying causes, and potential solutions. Our study results show that (1) Model Issue is the most common issue faced by practitioners, (2) Model Problem, Configuration and Connection Problem, and Feature and Method Problem are identified as the most frequent causes of the issues, and (3) Optimize Model is the predominant solution to the issues. Based on the study results, we provide implications for practitioners and researchers of LLM open-source projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v1">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and non-transferable across LLMs or tasks. Therefore, this paper proposes AutoPDL, an automated approach to discover good LLM agent configurations. Our method frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and six LLMs (ranging from 8B to 70B parameters) show consistent accuracy gains ($9.5\pm17.5$ percentage points), up to 68.9pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20177v4">AutoScale: Scale-Aware Data Mixing for Pre-Training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Domain reweighting is an emerging research area aimed at adjusting the relative weights of different data sources to improve the effectiveness and efficiency of LLM pre-training. We show that data mixtures that perform well at smaller scales may not retain their advantage at larger scales, challenging the existing practice of determining competitive mixtures in small-scale experiments and directly applying them at much larger scales. To address this, we propose AutoScale, a two-stage, scale-aware data composition framework. First, AutoScale fits a parametric model that predicts the model's loss under different data compositions, then uses it to find an approximate best allocation at smaller, more manageable budgets. Next, leveraging a novel theoretical analysis of how optimal compositions evolve with scale, AutoScale extrapolates that composition to larger budgets without further retraining. Empirically, AutoScale accelerates convergence and improves downstream performance. For instance, when pre-training GPT-2 Large, it achieves a 28% faster perplexity reduction than baselines and up to a 38% speed-up over unweighted training, while yielding best-average results on various downstream tasks. Overall, our findings illustrate how domain importance shifts with training scale, underscoring the need for scale-dependent data curation in LLM training. Our code is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04314v1">Balancing Complexity and Informativeness in LLM-Based Clustering: Finding the Goldilocks Zone</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 12 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The challenge of clustering short text data lies in balancing informativeness with interpretability. Traditional evaluation metrics often overlook this trade-off. Inspired by linguistic principles of communicative efficiency, this paper investigates the optimal number of clusters by quantifying the trade-off between informativeness and cognitive simplicity. We use large language models (LLMs) to generate cluster names and evaluate their effectiveness through semantic density, information theory, and clustering accuracy. Our results show that Gaussian Mixture Model (GMM) clustering on embeddings generated by a LLM, increases semantic density compared to random assignment, effectively grouping similar bios. However, as clusters increase, interpretability declines, as measured by a generative LLM's ability to correctly assign bios based on cluster names. A logistic regression analysis confirms that classification accuracy depends on the semantic similarity between bios and their assigned cluster names, as well as their distinction from alternatives. These findings reveal a "Goldilocks zone" where clusters remain distinct yet interpretable. We identify an optimal range of 16-22 clusters, paralleling linguistic efficiency in lexical categorization. These insights inform both theoretical models and practical applications, guiding future research toward optimising cluster interpretability and usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17238v3">IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      Software is prone to security vulnerabilities. Program analysis tools to detect them have limited effectiveness in practice due to their reliance on human labeled specifications. Large language models (or LLMs) have shown impressive code generation capabilities but they cannot do complex reasoning over code to detect such vulnerabilities especially since this task requires whole-repository analysis. We propose IRIS, a neuro-symbolic approach that systematically combines LLMs with static analysis to perform whole-repository reasoning for security vulnerability detection. Specifically, IRIS leverages LLMs to infer taint specifications and perform contextual analysis, alleviating needs for human specifications and inspection. For evaluation, we curate a new dataset, CWE-Bench-Java, comprising 120 manually validated security vulnerabilities in real-world Java projects. A state-of-the-art static analysis tool CodeQL detects only 27 of these vulnerabilities whereas IRIS with GPT-4 detects 55 (+28) and improves upon CodeQL's average false discovery rate by 5% points. Furthermore, IRIS identifies 4 previously unknown vulnerabilities which cannot be found by existing tools. IRIS is available publicly at https://github.com/iris-sast/iris.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21934v2">Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce the first comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15549v3">WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, aligning these models with human preferences has emerged as a critical challenge. Traditional alignment methods, relying on human or LLM annotated datasets, are limited by their resource-intensive nature, inherent subjectivity, misalignment with real-world user preferences, and the risk of feedback loops that amplify model biases. To overcome these limitations, we introduce WildFeedback, a novel framework that leverages in-situ user feedback during conversations with LLMs to create preference datasets automatically. Given a corpus of multi-turn user-LLM conversation, WildFeedback identifies and classifies user feedback to LLM responses between conversation turns. The user feedback is then used to create examples of preferred and dispreferred responses according to users' preference. Our experiments demonstrate that LLMs fine-tuned on WildFeedback dataset exhibit significantly improved alignment with user preferences, as evidenced by both traditional benchmarks and our proposed checklist-guided evaluation. By incorporating in-situ feedback from actual users, WildFeedback addresses the scalability, subjectivity, and bias challenges that plague existing approaches, marking a significant step toward developing LLMs that are more responsive to the diverse and evolving needs of their users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04524v1">Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 10pages
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have rapidly evolved, approaching Artificial General Intelligence (AGI) while benefiting from large-scale reinforcement learning to enhance Human Alignment (HA) and Reasoning. Recent reward-based optimization algorithms, such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) have achieved significant performance on reasoning tasks, whereas preference-based optimization algorithms such as Direct Preference Optimization (DPO) significantly improve the performance of LLMs on human alignment. However, despite the strong performance of reward-based optimization methods in alignment tasks , they remain vulnerable to reward hacking. Furthermore, preference-based algorithms (such as Online DPO) haven't yet matched the performance of reward-based optimization algorithms (like PPO) on reasoning tasks, making their exploration in this specific area still a worthwhile pursuit. Motivated by these challenges, we propose the Trust Region Preference Approximation (TRPA) algorithm, which integrates rule-based optimization with preference-based optimization for reasoning tasks. As a preference-based algorithm, TRPA naturally eliminates the reward hacking issue. TRPA constructs preference levels using predefined rules, forms corresponding preference pairs, and leverages a novel optimization algorithm for RL training with a theoretical monotonic improvement guarantee. Experimental results demonstrate that TRPA not only achieves competitive performance on reasoning tasks but also exhibits robust stability. The code of this paper are released and updating on https://github.com/XueruiSu/Trust-Region-Preference-Approximation.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01466v2">Enhancing LLM-Based Text Classification in Political Science: Automatic Prompt Optimization and Dynamic Exemplar Selection for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 46 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer substantial promise for text classification in political science, yet their effectiveness often depends on high-quality prompts and exemplars. To address this, we introduce a three-stage framework that enhances LLM performance through automatic prompt optimization, dynamic exemplar selection, and a consensus mechanism. Our approach automates prompt refinement using task-specific exemplars, eliminating speculative trial-and-error adjustments and producing structured prompts aligned with human-defined criteria. In the second stage, we dynamically select the most relevant exemplars, ensuring contextually appropriate guidance for each query. Finally, our consensus mechanism mimics the role of multiple human coders for a single task, combining outputs from LLMs to achieve high reliability and consistency at a reduced cost. Evaluated across tasks including sentiment analysis, stance detection, and campaign ad tone classification, our method enhances classification accuracy without requiring task-specific model retraining or extensive manual adjustments to prompts. This framework not only boosts accuracy, interpretability and transparency but also provides a cost-effective, scalable solution tailored to political science applications. An open-source Python package (PoliPrompt) is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06681v2">Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 39 pages; updated framework, literature review, and results
    </div>
    <details class="paper-abstract">
      In transportation system demand modeling and simulation, agent-based models and microsimulations are current state-of-the-art approaches. However, existing agent-based models still have some limitations on behavioral realism and resource demand that limit their applicability. In this study, leveraging the emerging technology of large language models (LLMs) and LLM-based agents, we propose a general LLM-agent-based modeling framework for transportation systems. We argue that LLM agents not only possess the essential capabilities to function as agents but also offer promising solutions to overcome some limitations of existing agent-based models. Our conceptual framework design closely replicates the decision-making and interaction processes and traits of human travelers within transportation networks, and we demonstrate that the proposed systems can meet critical behavioral criteria for decision-making and learning behaviors using related studies and a demonstrative example of LLM agents' learning and adjustment in the bottleneck setting. Although further refinement of the LLM-agent-based modeling framework is necessary, we believe that this approach has the potential to improve transportation system modeling and simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04485v1">Building LLM Agents by Incorporating Insights from Computer Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      LLM-driven autonomous agents have emerged as a promising direction in recent years. However, many of these LLM agents are designed empirically or based on intuition, often lacking systematic design principles, which results in diverse agent structures with limited generality and scalability. In this paper, we advocate for building LLM agents by incorporating insights from computer systems. Inspired by the von Neumann architecture, we propose a structured framework for LLM agentic systems, emphasizing modular design and universal principles. Specifically, this paper first provides a comprehensive review of LLM agents from the computer system perspective, then identifies key challenges and future directions inspired by computer system design, and finally explores the learning mechanisms for LLM agents beyond the computer system. The insights gained from this comparative analysis offer a foundation for systematic LLM agent design and advancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04471v1">VideoAgent2: Enhancing the LLM-Based Agent System for Long-Form Video Understanding by Uncertainty-Aware CoT</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      Long video understanding has emerged as an increasingly important yet challenging task in computer vision. Agent-based approaches are gaining popularity for processing long videos, as they can handle extended sequences and integrate various tools to capture fine-grained information. However, existing methods still face several challenges: (1) they often rely solely on the reasoning ability of large language models (LLMs) without dedicated mechanisms to enhance reasoning in long video scenarios; and (2) they remain vulnerable to errors or noise from external tools. To address these issues, we propose a specialized chain-of-thought (CoT) process tailored for long video analysis. Our proposed CoT with plan-adjust mode enables the LLM to incrementally plan and adapt its information-gathering strategy. We further incorporate heuristic uncertainty estimation of both the LLM and external tools to guide the CoT process. This allows the LLM to assess the reliability of newly collected information, refine its collection strategy, and make more robust decisions when synthesizing final answers. Empirical experiments show that our uncertainty-aware CoT effectively mitigates noise from external tools, leading to more reliable outputs. We implement our approach in a system called VideoAgent2, which also includes additional modules such as general context acquisition and specialized tool design. Evaluation on three dedicated long video benchmarks (and their subsets) demonstrates that VideoAgent2 outperforms the previous state-of-the-art agent-based method, VideoAgent, by an average of 13.1% and achieves leading performance among all zero-shot approaches
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04462v1">An overview of model uncertainty and variability in LLM-based sentiment analysis. Challenges, mitigation strategies and the role of explainability</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 25 pages and 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced sentiment analysis, yet their inherent uncertainty and variability pose critical challenges to achieving reliable and consistent outcomes. This paper systematically explores the Model Variability Problem (MVP) in LLM-based sentiment analysis, characterized by inconsistent sentiment classification, polarization, and uncertainty arising from stochastic inference mechanisms, prompt sensitivity, and biases in training data. We analyze the core causes of MVP, presenting illustrative examples and a case study to highlight its impact. In addition, we investigate key challenges and mitigation strategies, paying particular attention to the role of temperature as a driver of output randomness and emphasizing the crucial role of explainability in improving transparency and user trust. By providing a structured perspective on stability, reproducibility, and trustworthiness, this study helps develop more reliable, explainable, and robust sentiment analysis models, facilitating their deployment in high-stakes domains such as finance, healthcare, and policymaking, among others.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10714v2">ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-06
    </div>
    <details class="paper-abstract">
      The linear growth of key-value (KV) cache memory and quadratic computational in attention mechanisms complexity pose significant bottlenecks for large language models (LLMs) in long-context processing. While existing KV cache optimization methods address these challenges through token pruning or feature merging, they often incur irreversible information loss or require costly parameter retraining. To this end, we propose ZSMerge, a dynamic KV cache compression framework designed for efficient cache management, featuring three key operations: (1) fine-grained memory allocation guided by multi-dimensional token importance metrics at head-level granularity, (2) a residual merging mechanism that preserves critical context through compensated attention scoring, and (3) a zero-shot adaptation mechanism compatible with diverse LLM architectures without requiring retraining. ZSMerge significantly enhances memory efficiency and inference speed with negligible performance degradation across LLMs. When applied to LLaMA2-7B, it demonstrates a 20:1 compression ratio for key-value cache retention (reducing memory footprint to 5\% of baseline) while sustaining comparable generation quality, coupled with triple throughput gains at extreme 54k-token contexts that eliminate out-of-memory failures. The code is available at https://github.com/SusCom-Lab/ZSMerge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04429v1">IntentContinuum: Using LLMs to Support Intent-Based Computing Across the Compute Continuum</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      The increasing proliferation of IoT devices and AI applications has created a demand for scalable and efficient computing solutions, particularly for applications requiring real-time processing. The compute continuum integrates edge and cloud resources to meet this need, balancing the low-latency demands of the edge with the high computational power of the cloud. However, managing resources in such a distributed environment presents challenges due to the diversity and complexity of these systems. Traditional resource management methods, often relying on heuristic algorithms, struggle to manage the increasing complexity, scale, and dynamics of these systems, as well as adapt to dynamic workloads and changing network conditions. Moreover, designing such approaches is often time-intensive and highly tailored to specific applications, demanding deep expertise. In this paper, we introduce a novel framework for intent-driven resource management in the compute continuum, using large language models (LLMs) to help automate decision-making processes. Our framework ensures that user-defined intents -- such as achieving the required response times for time-critical applications -- are consistently fulfilled. In the event of an intent violation, our system performs root cause analysis by examining system data to identify and address issues. This approach reduces the need for human intervention and enhances system reliability, offering a more dynamic and efficient solution for resource management in distributed environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17867v3">Evaluating and Enhancing LLMs for Multi-turn Text-to-SQL with Multiple Question Types</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 International Joint Conference on Neural Networks 2025 (IJCNN 2025)
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly advanced text-to-SQL systems. However, most LLM-based methods often narrowly focus on SQL generation, neglecting the complexities of real-world conversational queries. This oversight can lead to unreliable responses, particularly for ambiguous questions that cannot be directly addressed with SQL. To bridge this gap, we propose MMSQL, a comprehensive test suite designed to evaluate the question classification and SQL generation capabilities of LLMs by simulating real-world scenarios with diverse question types and multi-turn Q&A interactions. Using MMSQL, we assessed the performance of popular LLMs, including both open-source and closed-source models, and identified key factors impacting their performance in such scenarios. Moreover, we introduce an LLM-based multi-agent framework that employs specialized agents to identify question types and determine appropriate answering strategies. Our experiments demonstrate that this approach significantly enhances the model's ability to navigate the complexities of conversational dynamics, effectively handling the diverse and complex nature of user queries. Our dataset and code are publicly available at https://mcxiaoxiao.github.io/MMSQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04386v1">Decoding Recommendation Behaviors of In-Context Learning LLMs Through Gradient Descent</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing trend in utilizing large language models (LLMs) for recommender systems, referred to as LLMRec. A notable approach within this trend is not to fine-tune these models directly but instead to leverage In-Context Learning (ICL) methods tailored for LLMRec, denoted as LLM-ICL Rec. Many contemporary techniques focus on harnessing ICL content to enhance LLMRec performance. However, optimizing LLMRec with ICL content presents unresolved challenges. Specifically, two key issues stand out: (1) the limited understanding of why using a few demonstrations without model fine-tuning can lead to better performance compared to zero-shot recommendations. (2) the lack of evaluation metrics for demonstrations in LLM-ICL Rec and the absence of the theoretical analysis and practical design for optimizing the generation of ICL content for recommendation contexts. To address these two main issues, we propose a theoretical model, the LLM-ICL Recommendation Equivalent Gradient Descent model (LRGD) in this paper, which connects recommendation generation with gradient descent dynamics. We demonstrate that the ICL inference process in LLM aligns with the training procedure of its dual model, producing token predictions equivalent to the dual model's testing outputs. Building on these theoretical insights, we propose an evaluation metric for assessing demonstration quality. We integrate perturbations and regularizations in LRGD to enhance the robustness of the recommender system. To further improve demonstration effectiveness, prevent performance collapse, and ensure long-term adaptability, we also propose a two-stage optimization process in practice. Extensive experiments and detailed analysis on three Amazon datasets validate the theoretical equivalence and support the effectiveness of our theoretical analysis and practical module design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v2">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-04-06
      | 💬 Appearing at VerifAI@ICLR 2025
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07920v4">Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted at ECIR'25
    </div>
    <details class="paper-abstract">
      Cross-encoders distilled from large language models (LLMs) are often more effective re-rankers than cross-encoders fine-tuned on manually labeled data. However, distilled models do not match the effectiveness of their teacher LLMs. We hypothesize that this effectiveness gap is due to the fact that previous work has not applied the best-suited methods for fine-tuning cross-encoders on manually labeled data (e.g., hard-negative sampling, deep sampling, and listwise loss functions). To close this gap, we create a new dataset, Rank-DistiLLM. Cross-encoders trained on Rank-DistiLLM achieve the effectiveness of LLMs while being up to 173 times faster and 24 times more memory efficient. Our code and data is available at https://github.com/webis-de/ECIR-25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15860v2">Synthetic vs. Gold: The Role of LLM-Generated Labels and Data in Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Cyberbullying (CB) presents a pressing threat, especially to children, underscoring the urgent need for robust detection systems to ensure online safety. However, progress in developing such systems is hindered by the scarcity of large, labeled datasets that are specifically tailored for specialized tasks and the target age groups. Creating these datasets relies heavily on human annotation, which not only strains resources but also raises significant ethical and legal concerns due to annotators' exposure to harmful content, notwithstanding the acquisition of this type of data from vulnerable populations such as children. In this paper, we address these challenges by leveraging Large Language Models (LLMs) to generate synthetic data and labels. Our experiments demonstrate that synthetic data enables BERT-based CB classifiers to achieve performance close to that of those trained on fully authentic datasets (75.8% vs. 81.5% accuracy). Additionally, LLMs can effectively label authentic yet unlabeled data, allowing BERT classifiers to attain a comparable performance level (79.1% vs. 81.5% accuracy). These results highlight the potential of LLMs as a scalable, ethical, and cost-effective solution for generating data for CB detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04110v1">PEIRCE: Unifying Material and Formal Reasoning via LLM-Driven Neuro-Symbolic Refinement</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Demo paper. Work in progress
    </div>
    <details class="paper-abstract">
      A persistent challenge in AI is the effective integration of material and formal inference - the former concerning the plausibility and contextual relevance of arguments, while the latter focusing on their logical and structural validity. Large Language Models (LLMs), by virtue of their extensive pre-training on large textual corpora, exhibit strong capabilities in material inference. However, their reasoning often lacks formal rigour and verifiability. At the same time, LLMs' linguistic competence positions them as a promising bridge between natural and formal languages, opening up new opportunities for combining these two modes of reasoning. In this paper, we introduce PEIRCE, a neuro-symbolic framework designed to unify material and formal inference through an iterative conjecture-criticism process. Within this framework, LLMs play the central role of generating candidate solutions in natural and formal languages, which are then evaluated and refined via interaction with external critique models. These critiques include symbolic provers, which assess formal validity, as well as soft evaluators that measure the quality of the generated arguments along linguistic and epistemic dimensions such as plausibility, coherence, and parsimony. While PEIRCE is a general-purpose framework, we demonstrate its capabilities in the domain of natural language explanation generation - a setting that inherently demands both material adequacy and formal correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03747v2">Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 no comment
    </div>
    <details class="paper-abstract">
      Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment but overlook LLMs' inherent strength on natural language processing -- their deep understanding of linguistic logic and structure rather than superficial embedding processing. We propose Context-Alignment, a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by a Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Demonstration examples prompt are employed to construct Demonstration Examples based Context-Alignment (DECA) following DSCA-GNNs framework. DECA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of DECA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provide powerful prior knowledge on context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04083v1">A Benchmark for End-to-End Zero-Shot Biomedical Relation Extraction with LLMs: Experiments with OpenAI Models</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Objective: Zero-shot methodology promises to cut down on costs of dataset annotation and domain expertise needed to make use of NLP. Generative large language models trained to align with human goals have achieved high zero-shot performance across a wide variety of tasks. As of yet, it is unclear how well these models perform on biomedical relation extraction (RE). To address this knowledge gap, we explore patterns in the performance of OpenAI LLMs across a diverse sampling of RE tasks. Methods: We use OpenAI GPT-4-turbo and their reasoning model o1 to conduct end-to-end RE experiments on seven datasets. We use the JSON generation capabilities of GPT models to generate structured output in two ways: (1) by defining an explicit schema describing the structure of relations, and (2) using a setting that infers the structure from the prompt language. Results: Our work is the first to study and compare the performance of the GPT-4 and o1 for the end-to-end zero-shot biomedical RE task across a broad array of datasets. We found the zero-shot performances to be proximal to that of fine-tuned methods. The limitations of this approach are that it performs poorly on instances containing many relations and errs on the boundaries of textual mentions. Conclusion: Recent large language models exhibit promising zero-shot capabilities in complex biomedical RE tasks, offering competitive performance with reduced dataset curation and NLP modeling needs at the cost of increased computing, potentially increasing medical community accessibility. Addressing the limitations we identify could further boost reliability. The code, data, and prompts for all our experiments are publicly available: https://github.com/bionlproc/ZeroShotRE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04060v1">VocalNet: Speech LLM with Multi-Token Prediction for Faster and High-Quality Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Speech large language models (LLMs) have emerged as a prominent research focus in speech processing. We propose VocalNet-1B and VocalNet-8B, a series of high-performance, low-latency speech LLMs enabled by a scalable and model-agnostic training framework for real-time voice interaction. Departing from the conventional next-token prediction (NTP), we introduce multi-token prediction (MTP), a novel approach optimized for speech LLMs that simultaneously improves generation speed and quality. Experiments show that VocalNet outperforms mainstream Omni LLMs despite using significantly less training data, while also surpassing existing open-source speech LLMs by a substantial margin. To support reproducibility and community advancement, we will open-source all model weights, inference code, training data, and framework implementations upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04030v1">OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v3">Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12561v3">UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental yet challenging (e.g., inflexible to iterate the study design flaws and hard to recruit study participants) research method for user experience (UX) researchers to evaluate a web design. Recent advances in Large Language Model-simulated Agent (LLM-Agent) research inspired us to design UXAgent to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human subject study. Our system features an LLM-Agent module and a universal browser connector module so that UX researchers can automatically generate thousands of simulated users to test the target website. The results are shown in qualitative (e.g., interviewing how an agent thinks ), quantitative (e.g., # of actions), and video recording formats for UX researchers to analyze. Through a heuristic user evaluation with five UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent-assisted UX study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12784v2">JudgeBench: A Benchmark for Evaluating LLM-based Judges</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      LLM-based judges have emerged as a scalable alternative to human evaluation and are increasingly used to assess, compare, and improve models. However, the reliability of LLM-based judges themselves is rarely scrutinized. As LLMs become more advanced, their responses grow more sophisticated, requiring stronger judges to evaluate them. Existing benchmarks primarily focus on a judge's alignment with human preferences, but often fail to account for more challenging tasks where crowdsourced human preference is a poor indicator of factual and logical correctness. To address this, we propose a novel evaluation framework to objectively evaluate LLM-based judges. Based on this framework, we propose JudgeBench, a benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding. JudgeBench leverages a novel pipeline for converting existing difficult datasets into challenging response pairs with preference labels reflecting objective correctness. Our comprehensive evaluation on a collection of prompted judges, fine-tuned judges, multi-agent judges, and reward models shows that JudgeBench poses a significantly greater challenge than previous benchmarks, with many strong models (e.g., GPT-4o) performing just slightly better than random guessing. Overall, JudgeBench offers a reliable platform for assessing increasingly advanced LLM-based judges. Data and code are available at https://github.com/ScalerLab/JudgeBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07983v1">Psychological Health Knowledge-Enhanced LLM-based Social Network Crisis Intervention Text Transfer Recognition Method</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      As the prevalence of mental health crises increases on social media platforms, identifying and preventing potential harm has become an urgent challenge. This study introduces a large language model (LLM)-based text transfer recognition method for social network crisis intervention, enhanced with domain-specific mental health knowledge. We propose a multi-level framework that incorporates transfer learning using BERT, and integrates mental health knowledge, sentiment analysis, and behavior prediction techniques. The framework includes a crisis annotation tool trained on social media datasets from real-world events, enabling the model to detect nuanced emotional cues and identify psychological crises. Experimental results show that the proposed method outperforms traditional models in crisis detection accuracy and exhibits greater sensitivity to subtle emotional and contextual variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04295v1">Dynamic Hedging Strategies in Derivatives Markets with LLM-Driven Sentiment and News Analytics</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Dynamic hedging strategies are essential for effective risk management in derivatives markets, where volatility and market sentiment can greatly impact performance. This paper introduces a novel framework that leverages large language models (LLMs) for sentiment analysis and news analytics to inform hedging decisions. By analyzing textual data from diverse sources like news articles, social media, and financial reports, our approach captures critical sentiment indicators that reflect current market conditions. The framework allows for real-time adjustments to hedging strategies, adapting positions based on continuous sentiment signals. Backtesting results on historical derivatives data reveal that our dynamic hedging strategies achieve superior risk-adjusted returns compared to conventional static approaches. The incorporation of LLM-driven sentiment analysis into hedging practices presents a significant advancement in decision-making processes within derivatives trading. This research showcases how sentiment-informed dynamic hedging can enhance portfolio management and effectively mitigate associated risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04292v1">Cross-Asset Risk Management: Integrating LLMs for Real-Time Monitoring of Equity, Fixed Income, and Currency Markets</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted by IJCNN 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as powerful tools in the field of finance, particularly for risk management across different asset classes. In this work, we introduce a Cross-Asset Risk Management framework that utilizes LLMs to facilitate real-time monitoring of equity, fixed income, and currency markets. This innovative approach enables dynamic risk assessment by aggregating diverse data sources, ultimately enhancing decision-making processes. Our model effectively synthesizes and analyzes market signals to identify potential risks and opportunities while providing a holistic view of asset classes. By employing advanced analytics, we leverage LLMs to interpret financial texts, news articles, and market reports, ensuring that risks are contextualized within broader market narratives. Extensive backtesting and real-time simulations validate the framework, showing increased accuracy in predicting market shifts compared to conventional methods. The focus on real-time data integration enhances responsiveness, allowing financial institutions to manage risks adeptly under varying market conditions and promoting financial stability through the advanced application of LLMs in risk analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v4">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04199v1">Investigating and Mitigating Stereotype-aware Unfairness in LLM-based Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated unprecedented language understanding and reasoning capabilities to capture diverse user preferences and advance personalized recommendations. Despite the growing interest in LLM-based personalized recommendations, unique challenges are brought to the trustworthiness of LLM-based recommender systems (LLM-RS), since LLMs are likely to inherit stereotypes that are embedded ubiquitously in word embeddings due to their training on large-scale uncurated datasets. This leads to LLM-RS exhibiting stereotypical linguistic associations between users and items. However, there remains a lack of studies investigating the simultaneous existence of stereotypes between users and items in LLM-RS. To bridge this gap, this study reveals a new variant of fairness between stereotype groups containing both users and items, to quantify discrimination against stereotypes in LLM-RS. Moreover, in this paper, to mitigate stereotype-aware unfairness in textual user and item information, we propose a novel framework (MoS), in which an insightful stereotype-wise routing strategy over multiple stereotype-relevant experts is designed to learn unbiased representations against different stereotypes in LLM- RS. Extensive experiments are conducted to analyze the influence of stereotype-aware fairness in LLM-RS and the effectiveness of our proposed methods, which consistently outperform competitive benchmarks under various fairness settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04193v1">AiReview: An Open Platform for Accelerating Systematic Reviews with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-05
      | 💬 Accepted at SIGIR 2025
    </div>
    <details class="paper-abstract">
      Systematic reviews are fundamental to evidence-based medicine. Creating one is time-consuming and labour-intensive, mainly due to the need to screen, or assess, many studies for inclusion in the review. Several tools have been developed to streamline this process, mostly relying on traditional machine learning methods. Large language models (LLMs) have shown potential in further accelerating the screening process. However, no tool currently allows end users to directly leverage LLMs for screening or facilitates systematic and transparent usage of LLM-assisted screening methods. This paper introduces (i) an extensible framework for applying LLMs to systematic review tasks, particularly title and abstract screening, and (ii) a web-based interface for LLM-assisted screening. Together, these elements form AiReview-a novel platform for LLM-assisted systematic review creation. AiReview is the first of its kind to bridge the gap between cutting-edge LLM-assisted screening methods and those that create medical systematic reviews. The tool is available at https://aireview.ielab.io. The source code is also open sourced at https://github.com/ielab/ai-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09334v2">CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04187v1">AttackLLM: LLM-based Attack Pattern Generation for an Industrial Control System</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Malicious examples are crucial for evaluating the robustness of machine learning algorithms under attack, particularly in Industrial Control Systems (ICS). However, collecting normal and attack data in ICS environments is challenging due to the scarcity of testbeds and the high cost of human expertise. Existing datasets are often limited by the domain expertise of practitioners, making the process costly and inefficient. The lack of comprehensive attack pattern data poses a significant problem for developing robust anomaly detection methods. In this paper, we propose a novel approach that combines data-centric and design-centric methodologies to generate attack patterns using large language models (LLMs). Our results demonstrate that the attack patterns generated by LLMs not only surpass the quality and quantity of those created by human experts but also offer a scalable solution that does not rely on expensive testbeds or pre-existing attack examples. This multi-agent based approach presents a promising avenue for enhancing the security and resilience of ICS environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04178v1">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals. To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at https://github.com/WANGBohaO-jpg/MSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04152v1">Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03360v1">Sustainable LLM Inference for Edge AI: Evaluating Quantized LLMs for Energy Efficiency, Output Accuracy, and Inference Latency</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 30 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices presents significant challenges due to computational constraints, memory limitations, inference speed, and energy consumption. Model quantization has emerged as a key technique to enable efficient LLM inference by reducing model size and computational overhead. In this study, we conduct a comprehensive analysis of 28 quantized LLMs from the Ollama library, which applies by default Post-Training Quantization (PTQ) and weight-only quantization techniques, deployed on an edge device (Raspberry Pi 4 with 4GB RAM). We evaluate energy efficiency, inference performance, and output accuracy across multiple quantization levels and task types. Models are benchmarked on five standardized datasets (CommonsenseQA, BIG-Bench Hard, TruthfulQA, GSM8K, and HumanEval), and we employ a high-resolution, hardware-based energy measurement tool to capture real-world power consumption. Our findings reveal the trade-offs between energy efficiency, inference speed, and accuracy in different quantization settings, highlighting configurations that optimize LLM deployment for resource-constrained environments. By integrating hardware-level energy profiling with LLM benchmarking, this study provides actionable insights for sustainable AI, bridging a critical gap in existing research on energy-aware LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03343v1">Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00159v3">LLMs Prompted for Graphs: Hallucinations and Generative Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 A preliminary version of this work appeared in the Complex Networks 2024 conference, under the title "LLMs hallucinate graphs too: a structural perspective"
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are nowadays prompted for a wide variety of tasks. In this article, we investigate their ability in reciting and generating graphs. We first study the ability of LLMs to regurgitate well known graphs from the literature (e.g. Karate club or the graph atlas)4. Secondly, we question the generative capabilities of LLMs by asking for Erdos-Renyi random graphs. As opposed to the possibility that they could memorize some Erdos-Renyi graphs included in their scraped training set, this second investigation aims at studying a possible emergent property of LLMs. For both tasks, we propose a metric to assess their errors with the lens of hallucination (i.e. incorrect information returned as facts). We most notably find that the amplitude of graph hallucinations can characterize the superiority of some LLMs. Indeed, for the recitation task, we observe that graph hallucinations correlate with the Hallucination Leaderboard, a hallucination rank that leverages 10, 000 times more prompts to obtain its ranking. For the generation task, we find surprisingly good and reproducible results in most of LLMs. We believe this to constitute a starting point for more in-depth studies of this emergent capability and a challenging benchmark for their improvements. Altogether, these two aspects of LLMs capabilities bridge a gap between the network science and machine learning communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03312v1">Evaluating Compact LLMs for Zero-Shot Iberian Language Tasks on End-User Devices</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Under Revision al SEPLN conference
    </div>
    <details class="paper-abstract">
      Large Language Models have significantly advanced natural language processing, achieving remarkable performance in tasks such as language generation, translation, and reasoning. However, their substantial computational requirements restrict deployment to high-end systems, limiting accessibility on consumer-grade devices. This challenge is especially pronounced for under-resourced languages like those spoken in the Iberian Peninsula, where relatively limited linguistic resources and benchmarks hinder effective evaluation. This work presents a comprehensive evaluation of compact state-of-the-art LLMs across several essential NLP tasks tailored for Iberian languages. The results reveal that while some models consistently excel in certain tasks, significant performance gaps remain, particularly for languages such as Basque. These findings highlight the need for further research on balancing model compactness with robust multilingual performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03255v1">Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 12 pages content (incl. appendix) + 12 pages references, comments welcome
    </div>
    <details class="paper-abstract">
      Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention over effective governance policies, monitoring and control protocols. Based on emerging landscapes of the agentic market, we analyze the potential liability issues stemming from delegated use of LLM agents and their extended systems from a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing and monitoring approaches to enhancing transparency and accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v2">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03174v1">Multi-lingual Multi-turn Automated Red Teaming for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted at TrustNLP@NAACL 2025
    </div>
    <details class="paper-abstract">
      Language Model Models (LLMs) have improved dramatically in the past few years, increasing their adoption and the scope of their capabilities over time. A significant amount of work is dedicated to ``model alignment'', i.e., preventing LLMs to generate unsafe responses when deployed into customer-facing applications. One popular method to evaluate safety risks is \textit{red-teaming}, where agents attempt to bypass alignment by crafting elaborate prompts that trigger unsafe responses from a model. Standard human-driven red-teaming is costly, time-consuming and rarely covers all the recent features (e.g., multi-lingual, multi-modal aspects), while proposed automation methods only cover a small subset of LLMs capabilities (i.e., English or single-turn). We present Multi-lingual Multi-turn Automated Red Teaming (\textbf{MM-ART}), a method to fully automate conversational, multi-lingual red-teaming operations and quickly identify prompts leading to unsafe responses. Through extensive experiments on different languages, we show the studied LLMs are on average 71\% more vulnerable after a 5-turn conversation in English than after the initial turn. For conversations in non-English languages, models display up to 195\% more safety vulnerabilities than the standard single-turn English approach, confirming the need for automated red-teaming methods matching LLMs capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03111v1">Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03966v1">Bridging LMS and Generative AI: Dynamic Course Content Integration (DCCI) for Connecting LLMs to Course Content -- The Ask ME Assistant</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with Learning Management Systems (LMSs) has the potential to enhance task automation and accessibility in education. However, hallucination where LLMs generate inaccurate or misleading information remains a significant challenge. This study introduces the Dynamic Course Content Integration (DCCI) mechanism, which dynamically retrieves and integrates course content and curriculum from Canvas LMS into the LLM-powered assistant, Ask ME. By employing prompt engineering to structure retrieved content within the LLM's context window, DCCI ensures accuracy, relevance, and contextual alignment, mitigating hallucination. To evaluate DCCI's effectiveness, Ask ME's usability, and broader student perceptions of AI in education, a mixed-methods approach was employed, incorporating user satisfaction ratings and a structured survey. Results from a pilot study indicate high user satisfaction (4.614/5), with students recognizing Ask ME's ability to provide timely and contextually relevant responses for both administrative and course-related inquiries. Additionally, a majority of students agreed that Ask ME's integration with course content in Canvas LMS reduced platform-switching, improving usability, engagement, and comprehension. AI's role in reducing classroom hesitation and fostering self-directed learning and intellectual curiosity was also highlighted. Despite these benefits and positive perception of AI tools, concerns emerged regarding over-reliance on AI, accuracy limitations, and ethical issues such as plagiarism and reduced student-teacher interaction. These findings emphasize the need for strategic AI implementation, ethical safeguards, and a pedagogical framework that prioritizes human-AI collaboration over substitution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14827v2">Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      In a prompt injection attack, an attacker injects a prompt into the original one, aiming to make an LLM follow the injected prompt to perform an attacker-chosen task. Existing attacks primarily focus on how to blend the injected prompt into the original prompt without altering the LLM itself. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we show that an attacker can boost the success of prompt injection attacks by poisoning the LLM's alignment process. Specifically, we propose PoisonedAlign, a method to strategically create poisoned alignment samples. When even a small fraction of the alignment data is poisoned using our method, the aligned LLM becomes more vulnerable to prompt injection while maintaining its foundational capabilities. The code is available at https://github.com/Sadcardation/PoisonedAlign
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18921v2">From Blind Solvers to Logical Thinkers: Benchmarking LLMs' Logical Integrity on Faulty Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Consider the math problem: "Lily received 3 cookies from her best friend yesterday and ate 5 for breakfast. Today, her friend gave her 3 more cookies. How many cookies does Lily have now?" Many large language models (LLMs) in previous research approach this problem by calculating the answer "1" using the equation "3 - 5 + 3." However, from a human perspective, we recognize the inherent flaw in this problem: Lily cannot eat 5 cookies if she initially only had 3. This discrepancy prompts a key question: Are current LLMs merely Blind Solver that apply mathematical operations without deeper reasoning, or can they function as Logical Thinker capable of identifying logical inconsistencies? To explore this question, we propose a benchmark dataset, FaultyMath, which includes faulty math problems of rich diversity: i) multiple mathematical categories, e.g., algebra, geometry, number theory, etc., ii) varying levels of difficulty, and iii) different origins of faultiness -- ranging from violations of common sense and ambiguous statements to mathematical contradictions and more. We evaluate a broad spectrum of LLMs, including open-source, closed-source, and math-specialized models, using FaultyMath across three dimensions: (i) How accurately can the models detect faulty math problems without being explicitly prompted to do so? (ii) When provided with hints -- either correct or misleading -- about the validity of the problems, to what extent do LLMs adapt to become reliable Logical Thinker? (iii) How trustworthy are the explanations generated by LLMs when they recognize a math problem as flawed? Through extensive experimentation and detailed analysis, our results demonstrate that existing LLMs largely function as Blind Solver and fall short of the reasoning capabilities required to perform as Logical Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18702v2">CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Retrieval from graph data is crucial for augmenting large language models (LLM) with both open-domain knowledge and private enterprise data, and it is also a key component in the recent GraphRAG system (edge et al., 2024). Despite decades of research on knowledge graphs and knowledge base question answering, leading LLM frameworks (e.g. Langchain and LlamaIndex) have only minimal support for retrieval from modern encyclopedic knowledge graphs like Wikidata. In this paper, we analyze the root cause and suggest that modern RDF knowledge graphs (e.g. Wikidata, Freebase) are less efficient for LLMs due to overly large schemas that far exceed the typical LLM context window, use of resource identifiers, overlapping relation types and lack of normalization. As a solution, we propose property graph views on top of the underlying RDF graph that can be efficiently queried by LLMs using Cypher. We instantiated this idea on Wikidata and introduced CypherBench, the first benchmark with 11 large-scale, multi-domain property graphs with 7.8 million entities and over 10,000 questions. To achieve this, we tackled several key challenges, including developing an RDF-to-property graph conversion engine, creating a systematic pipeline for text-to-Cypher task generation, and designing new evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03889v1">Using Attention Sinks to Identify and Evaluate Dormant Heads in Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Multi-head attention is foundational to large language models (LLMs), enabling different heads to have diverse focus on relevant input tokens. However, learned behaviors like attention sinks, where the first token receives most attention despite limited semantic importance, challenge our understanding of multi-head attention. To analyze this phenomenon, we propose a new definition for attention heads dominated by attention sinks, known as dormant attention heads. We compare our definition to prior work in a model intervention study where we test whether dormant heads matter for inference by zeroing out the output of dormant attention heads. Using six pretrained models and five benchmark datasets, we find our definition to be more model and dataset-agnostic. Using our definition on most models, more than 4% of a model's attention heads can be zeroed while maintaining average accuracy, and zeroing more than 14% of a model's attention heads can keep accuracy to within 1% of the pretrained model's average accuracy. Further analysis reveals that dormant heads emerge early in pretraining and can transition between dormant and active states during pretraining. Additionally, we provide evidence that they depend on characteristics of the input text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02559v2">Leveraging LLM For Synchronizing Information Across Multilingual Tables</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 17 Pages, 11 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03877v1">Concept-based Rubrics Improve LLM Formative Assessment and Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 13 pages excluding references. 9 tables and 4 figures
    </div>
    <details class="paper-abstract">
      Formative assessment in STEM topics aims to promote student learning by identifying students' current understanding, thus targeting how to promote further learning. Previous studies suggest that the assessment performance of current generative large language models (LLMs) on constructed responses to open-ended questions is significantly lower than that of supervised classifiers trained on high-quality labeled data. However, we demonstrate that concept-based rubrics can significantly enhance LLM performance, which narrows the gap between LLMs as off-the shelf assessment tools, and smaller supervised models, which need large amounts of training data. For datasets where concept-based rubrics allow LLMs to achieve strong performance, we show that the concept-based rubrics help the same LLMs generate high quality synthetic data for training lightweight, high-performance supervised models. Our experiments span diverse STEM student response datasets with labels of varying quality, including a new real-world dataset that contains some AI-assisted responses, which introduces additional considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00993v2">MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 18 pages, 11 figures, 6 tables. Project page: https://github.com/UCSC-VLAA/MedReason
    </div>
    <details class="paper-abstract">
      Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code is available at https://github.com/UCSC-VLAA/MedReason.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03846v1">Do LLM Evaluators Prefer Themselves for a Reason?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Preprint. 31 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference detrimental, or does it simply reflect objectively superior outputs from more capable models? Disentangling these has been challenging due to the usage of subjective tasks in previous studies. To address this, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful self-preference (favoring objectively worse responses) from legitimate self-preference (favoring genuinely superior ones). We conduct large-scale experiments under controlled evaluation conditions across diverse model families (e.g., Llama, Qwen, Gemma, Mistral, Phi, GPT, DeepSeek). Our findings reveal three key insights: (1) Better generators are better judges -- LLM evaluators' accuracy strongly correlates with their task performance, and much of the self-preference in capable models is legitimate. (2) Harmful self-preference persists, particularly when evaluator models perform poorly as generators on specific task instances. Stronger models exhibit more pronounced harmful bias when they err, though such incorrect generations are less frequent. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03822v1">Arti-"fickle" Intelligence: Using LLMs as a Tool for Inference in the Political and Social Sciences</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) are incredibly useful, versatile, and promising tools. However, they will be of most use to political and social science researchers when they are used in a way that advances understanding about real human behaviors and concerns. To promote the scientific use of LLMs, we suggest that researchers in the political and social sciences need to remain focused on the scientific goal of inference. To this end, we discuss the challenges and opportunities related to scientific inference with LLMs, using validation of model output as an illustrative case for discussion. We propose a set of guidelines related to establishing the failure and success of LLMs when completing particular tasks, and discuss how we can make inferences from these observations. We conclude with a discussion of how this refocus will improve the accumulation of shared scientific knowledge about these tools and their uses in the social sciences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v1">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly contributing to the creation of content on the Internet. This creates a feedback loop as subsequent generations of models will be trained on this generated, synthetic data. This phenomenon is receiving increasing interest, in particular because previous studies have shown that it may lead to distribution shift - models misrepresent and forget the true underlying distributions of human data they are expected to approximate (e.g. resulting in a drastic loss of quality). In this study, we study the impact of human data properties on distribution shift dynamics in iterated training loops. We first confirm that the distribution shift dynamics greatly vary depending on the human data by comparing four datasets (two based on Twitter and two on Reddit). We then test whether data quality may influence the rate of this shift. We find that it does on the twitter, but not on the Reddit datasets. We then focus on a Reddit dataset and conduct a more exhaustive evaluation of a large set of dataset properties. This experiment associated lexical diversity with larger, and semantic diversity with smaller detrimental shifts, suggesting that incorporating text with high lexical (but limited semantic) diversity could exacerbate the degradation of generated text. We then focus on the evolution of political bias, and find that the type of shift observed (bias reduction, amplification or inversion) depends on the political lean of the human (true) distribution. Overall, our work extends the existing literature on the consequences of recursive fine-tuning by showing that this phenomenon is highly dependent on features of the human data on which training occurs. This suggests that different parts of internet (e.g. GitHub, Reddit) may undergo different types of shift depending on their properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03598v1">EnrichIndex: Using LLMs to Enrich Retrieval Indices Offline</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Dataset and code are available at https://peterbaile.github.io/enrichindex/
    </div>
    <details class="paper-abstract">
      Existing information retrieval systems excel in cases where the language of target documents closely matches that of the user query. However, real-world retrieval systems are often required to implicitly reason whether a document is relevant. For example, when retrieving technical texts or tables, their relevance to the user query may be implied through a particular jargon or structure, rather than explicitly expressed in their content. Large language models (LLMs) hold great potential in identifying such implied relevance by leveraging their reasoning skills. Nevertheless, current LLM-augmented retrieval is hindered by high latency and computation cost, as the LLM typically computes the query-document relevance online, for every query anew. To tackle this issue we introduce EnrichIndex, a retrieval approach which instead uses the LLM offline to build semantically-enriched retrieval indices, by performing a single pass over all documents in the retrieval corpus once during ingestion time. Furthermore, the semantically-enriched indices can complement existing online retrieval approaches, boosting the performance of LLM re-rankers. We evaluated EnrichIndex on five retrieval tasks, involving passages and tables, and found that it outperforms strong online LLM-based retrieval systems, with an average improvement of 11.7 points in recall @ 10 and 10.6 points in NDCG @ 10 compared to strong baselines. In terms of online calls to the LLM, it processes 293.3 times fewer tokens which greatly reduces the online latency and cost. Overall, EnrichIndex is an effective way to build better retrieval indices offline by leveraging the strong reasoning skills of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03444v1">LLMSched: Uncertainty-Aware Workload Scheduling for Compound LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-04-04
    </div>
    <details class="paper-abstract">
      Developing compound Large Language Model (LLM) applications is becoming an increasingly prevalent approach to solving real-world problems. In these applications, an LLM collaborates with various external modules, including APIs and even other LLMs, to realize complex intelligent services. However, we reveal that the intrinsic duration and structural uncertainty in compound LLM applications pose great challenges for LLM service providers in serving and scheduling them efficiently. In this paper, we propose LLMSched, an uncertainty-aware scheduling framework for emerging compound LLM applications. In LLMSched, we first design a novel DAG-based model to describe the uncertain compound LLM applications. Then, we adopt the Bayesian network to comprehensively profile compound LLM applications and identify uncertainty-reducing stages, along with an entropy-based mechanism to quantify their uncertainty reduction. Combining an uncertainty reduction strategy and a job completion time (JCT)-efficient scheme, we further propose an efficient scheduler to reduce the average JCT. Evaluation of both simulation and testbed experiments on various representative compound LLM applications shows that compared to existing state-of-the-art scheduling schemes, LLMSched can reduce the average JCT by 14~79%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09893v2">RMB: Comprehensively Benchmarking Reward Models in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-04
      | 💬 Accepted by ICLR2025
    </div>
    <details class="paper-abstract">
      Reward models (RMs) guide the alignment of large language models (LLMs), steering them toward behaviors preferred by humans. Evaluating RMs is the key to better aligning LLMs. However, the current evaluation of RMs may not directly correspond to their alignment performance due to the limited distribution of evaluation data and evaluation methods that are not closely related to alignment objectives. To address these limitations, we propose RMB, a comprehensive RM benchmark that covers over 49 real-world scenarios and includes both pairwise and Best-of-N (BoN) evaluations to better reflect the effectiveness of RMs in guiding alignment optimization. We demonstrate a positive correlation between our benchmark and the downstream alignment task performance. Based on our benchmark, we conduct extensive analysis on the state-of-the-art RMs, revealing their generalization defects that were not discovered by previous benchmarks, and highlighting the potential of generative RMs. Furthermore, we delve into open questions in reward models, specifically examining the effectiveness of majority voting for the evaluation of reward models and analyzing the impact factors of generative RMs, including the influence of evaluation criteria and instructing methods. Our evaluation code and datasets are available at https://github.com/Zhou-Zoey/RMB-Reward-Model-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02708v1">The Hidden Space of Safety: Understanding Preference-Tuned LLMs in Multilingual context</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 14 pages, 11 Figures, 2 Tables, currently under review at ACL 2025
    </div>
    <details class="paper-abstract">
      Alignment tuning has enabled large language models to excel in reasoning, instruction-following, and minimizing harmful generations. However, despite their widespread deployment, these models exhibit a monolingual bias, raising concerns about the effectiveness of alignment across languages. Current alignment methods predominantly focus on English, leaving it unclear how alignment mechanism generalize to multilingual settings. To address this, we conduct a systematic analysis of distributional shifts in the embedding space of LLMs before and after alignment, uncovering its impact on model behavior across diverse languages. We leverage the alignment-induced separation in safety space as a quantitative tool to measure how alignment enforces safety constraints. Our study evaluates seven LLMs using balanced toxicity datasets and parallel text-detoxification benchmarks, revealing substantial disparities in the latent representation space between high-resource and low-resource languages. These findings underscore the need for language-specific fine-tuning to ensure fair, reliable and robust multilingual alignment. Our insights provide a foundation for developing truly safe multilingual LLMs, emphasizing the urgency of addressing alignment gaps in underrepresented languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02671v1">LLM for Complex Reasoning Task: An Exploratory Study in Fermi Problems</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 7 pages,7 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Fermi Problems (FPs) are mathematical reasoning tasks that require human-like logic and numerical reasoning. Unlike other reasoning questions, FPs often involve real-world impracticalities or ambiguous concepts, making them challenging even for humans to solve. Despite advancements in AI, particularly with large language models (LLMs) in various reasoning tasks, FPs remain relatively under-explored. This work conducted an exploratory study to examine the capabilities and limitations of LLMs in solving FPs. We first evaluated the overall performance of three advanced LLMs using a publicly available FP dataset. We designed prompts according to the recently proposed TELeR taxonomy, including a zero-shot scenario. Results indicated that all three LLMs achieved a fp_score (range between 0 - 1) below 0.5, underscoring the inherent difficulty of these reasoning tasks. To further investigate, we categorized FPs into standard and specific questions, hypothesizing that LLMs would perform better on standard questions, which are characterized by clarity and conciseness, than on specific ones. Comparative experiments confirmed this hypothesis, demonstrating that LLMs performed better on standard FPs in terms of both accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v1">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02622v1">Exploring undercurrents of learning tensions in an LLM-enhanced landscape: A student-centered qualitative perspective on LLM vs Search</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming how students learn by providing readily available tools that can quickly augment or complete various learning activities with non-trivial performance. Similar paradigm shifts have occurred in the past with the introduction of search engines and Wikipedia, which replaced or supplemented traditional information sources such as libraries and books. This study investigates the potential for LLMs to represent the next shift in learning, focusing on their role in information discovery and synthesis compared to existing technologies, such as search engines. Using a within-subjects, counterbalanced design, participants learned new topics using a search engine (Google) and an LLM (ChatGPT). Post-task follow-up interviews explored students' reflections, preferences, pain points, and overall perceptions. We present analysis of their responses that show nuanced insights into when, why, and how students prefer LLMs over search engines, offering implications for educators, policymakers, and technology developers navigating the evolving educational landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01380v2">Efficient LLM Inference using Dynamic Input Pruning and Cache-Aware Masking</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 Main Text: 10 pages, 11 figures. Appendix: 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      While mobile devices provide ever more compute power, improvements in DRAM bandwidth are much slower. This is unfortunate for large language model (LLM) token generation, which is heavily memory-bound. Previous work has proposed to leverage natural dynamic activation sparsity in ReLU-activated LLMs to reduce effective DRAM bandwidth per token. However, more recent LLMs use SwiGLU instead of ReLU, which results in little inherent sparsity. While SwiGLU activations can be pruned based on magnitude, the resulting sparsity patterns are difficult to predict, rendering previous approaches ineffective. To circumvent this issue, our work introduces Dynamic Input Pruning (DIP): a predictor-free dynamic sparsification approach, which preserves accuracy with minimal fine-tuning. DIP can further use lightweight LoRA adapters to regain some performance lost during sparsification. Lastly, we describe a novel cache-aware masking strategy, which considers the cache state and activation magnitude to further increase cache hit rate, improving LLM token rate on mobile devices. DIP outperforms other methods in terms of accuracy, memory and throughput trade-offs across simulated hardware settings. On Phi-3-Medium, DIP achieves a 46\% reduction in memory and 40\% increase in throughput with $<$ 0.1 loss in perplexity when compared to streaming the dense model from Flash. The open source code for HW simulator, methods, and experiments in this paper is available at https://github.com/Qualcomm-AI-research/dynamic-sparsity .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02559v1">Leveraging LLM For Synchronizing Information Across Multilingual Tables</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 17 Pages, 11 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      The vast amount of online information today poses challenges for non-English speakers, as much of it is concentrated in high-resource languages such as English and French. Wikipedia reflects this imbalance, with content in low-resource languages frequently outdated or incomplete. Recent research has sought to improve cross-language synchronization of Wikipedia tables using rule-based methods. These approaches can be effective, but they struggle with complexity and generalization. This paper explores large language models (LLMs) for multilingual information synchronization, using zero-shot prompting as a scalable solution. We introduce the Information Updation dataset, simulating the real-world process of updating outdated Wikipedia tables, and evaluate LLM performance. Our findings reveal that single-prompt approaches often produce suboptimal results, prompting us to introduce a task decomposition strategy that enhances coherence and accuracy. Our proposed method outperforms existing baselines, particularly in Information Updation (1.79%) and Information Addition (20.58%), highlighting the model strength in dynamically updating and enriching data across architectures
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02553v1">Exploring Individual Factors in the Adoption of LLMs for Specific Software Engineering Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) is transforming software development, significantly enhancing software engineering processes. Research has explored their role within development teams, focusing on specific tasks such as artifact generation, decision-making support, and information retrieval. Despite the growing body of work on LLMs in software engineering, most studies have centered on broad adoption trends, neglecting the nuanced relationship between individual cognitive and behavioral factors and their impact on task-specific adoption. While factors such as perceived effort and performance expectancy have been explored at a general level, their influence on distinct software engineering tasks remains underexamined. This gap hinders the development of tailored LLM-based systems (e.g., Generative AI Agents) that align with engineers' specific needs and limits the ability of team leaders to devise effective strategies for fostering LLM adoption in targeted workflows. This study bridges this gap by surveying N=188 software engineers to test the relationship between individual attributes related to technology adoption and LLM adoption across five key tasks, using structural equation modeling (SEM). The Unified Theory of Acceptance and Use of Technology (UTAUT2) was applied to characterize individual adoption behaviors. The findings reveal that task-specific adoption is influenced by distinct factors, some of which negatively impact adoption when considered in isolation, underscoring the complexity of LLM integration in software engineering. To support effective adoption, this article provides actionable recommendations, such as seamlessly integrating LLMs into existing development environments and encouraging peer-driven knowledge sharing to enhance information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02509v1">A Memory-Augmented LLM-Driven Method for Autonomous Merging of 3D Printing Work Orders</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 6 pages, 5 figures
    </div>
    <details class="paper-abstract">
      With the rapid development of 3D printing, the demand for personalized and customized production on the manufacturing line is steadily increasing. Efficient merging of printing workpieces can significantly enhance the processing efficiency of the production line. Addressing the challenge, a Large Language Model (LLM)-driven method is established in this paper for the autonomous merging of 3D printing work orders, integrated with a memory-augmented learning strategy. In industrial scenarios, both device and order features are modeled into LLM-readable natural language prompt templates, and develop an order-device matching tool along with a merging interference checking module. By incorporating a self-memory learning strategy, an intelligent agent for autonomous order merging is constructed, resulting in improved accuracy and precision in order allocation. The proposed method effectively leverages the strengths of LLMs in industrial applications while reducing hallucination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02507v1">ZClip: Adaptive Spike Mitigation for LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) presents numerous challenges, including gradient instability and loss spikes. These phenomena can lead to catastrophic divergence, requiring costly checkpoint restoration and data batch skipping. Traditional gradient clipping techniques, such as constant or norm-based methods, fail to address these issues effectively due to their reliance on fixed thresholds or heuristics, leading to inefficient learning and requiring frequent manual intervention. In this work, we propose ZClip, an adaptive gradient clipping algorithm that dynamically adjusts the clipping threshold based on statistical properties of gradient norms over time. Unlike prior reactive strategies, ZClip proactively adapts to training dynamics without making any prior assumptions on the scale and the temporal evolution of gradient norms. At its core, it leverages z-score-based anomaly detection to identify and mitigate large gradient spikes, preventing malignant loss spikes while not interfering with convergence otherwise. Our code is available at: https://github.com/bluorion-com/ZClip.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01667v2">Testing Low-Resource Language Support in LLMs Using Language Proficiency Exams: the Case of Luxembourgish</a></div>
    <div class="paper-meta">
      📅 2025-04-03
      | 💬 18 pages, 2 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an increasingly important tool in research and society at large. While LLMs are regularly used all over the world by experts and lay-people alike, they are predominantly developed with English-speaking users in mind, performing well in English and other wide-spread languages while less-resourced languages such as Luxembourgish are seen as a lower priority. This lack of attention is also reflected in the sparsity of available evaluation tools and datasets. In this study, we investigate the viability of language proficiency exams as such evaluation tools for the Luxembourgish language. We find that large models such as ChatGPT, Claude and DeepSeek-R1 typically achieve high scores, while smaller models show weak performances. We also find that the performances in such language exams can be used to predict performances in other NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02458v1">Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.
    </details>
</div>
