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
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- Part 13

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01714v1">Position: Towards a Responsible LLM-empowered Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The rise of Agent AI and Large Language Model-powered Multi-Agent Systems (LLM-MAS) has underscored the need for responsible and dependable system operation. Tools like LangChain and Retrieval-Augmented Generation have expanded LLM capabilities, enabling deeper integration into MAS through enhanced knowledge retrieval and reasoning. However, these advancements introduce critical challenges: LLM agents exhibit inherent unpredictability, and uncertainties in their outputs can compound across interactions, threatening system stability. To address these risks, a human-centered design approach with active dynamic moderation is essential. Such an approach enhances traditional passive oversight by facilitating coherent inter-agent communication and effective system governance, allowing MAS to achieve desired outcomes more efficiently.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18645v2">Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) leverage chain-of-thought (CoT) prompting to provide step-by-step rationales, improving performance on complex tasks. Despite its benefits, vanilla CoT often fails to fully verify intermediate inferences and can produce misleading explanations. In this work, we propose Layered Chain-of-Thought (Layered-CoT) Prompting, a novel framework that systematically segments the reasoning process into multiple layers, each subjected to external checks and optional user feedback. We expand on the key concepts, present three scenarios -- medical triage, financial risk assessment, and agile engineering -- and demonstrate how Layered-CoT surpasses vanilla CoT in terms of transparency, correctness, and user engagement. By integrating references from recent arXiv papers on interactive explainability, multi-agent frameworks, and agent-based collaboration, we illustrate how Layered-CoT paves the way for more reliable and grounded explanations in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01450v1">Simulating Rumor Spreading in Social Networks using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the rise of social media, misinformation has become increasingly prevalent, fueled largely by the spread of rumors. This study explores the use of Large Language Model (LLM) agents within a novel framework to simulate and analyze the dynamics of rumor propagation across social networks. To this end, we design a variety of LLM-based agent types and construct four distinct network structures to conduct these simulations. Our framework assesses the effectiveness of different network constructions and agent behaviors in influencing the spread of rumors. Our results demonstrate that the framework can simulate rumor spreading across more than one hundred agents in various networks with thousands of edges. The evaluations indicate that network structure, personas, and spreading schemes can significantly influence rumor dissemination, ranging from no spread to affecting 83\% of agents in iterations, thereby offering a realistic simulation of rumor spread in social networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01390v1">Plan-Then-Execute: An Empirical Study of User Trust and Team Performance When Using LLM Agents As A Daily Assistant</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 conditionally accepted to CHI 2025
    </div>
    <details class="paper-abstract">
      Since the explosion in popularity of ChatGPT, large language models (LLMs) have continued to impact our everyday lives. Equipped with external tools that are designed for a specific purpose (e.g., for flight booking or an alarm clock), LLM agents exercise an increasing capability to assist humans in their daily work. Although LLM agents have shown a promising blueprint as daily assistants, there is a limited understanding of how they can provide daily assistance based on planning and sequential decision making capabilities. We draw inspiration from recent work that has highlighted the value of 'LLM-modulo' setups in conjunction with humans-in-the-loop for planning tasks. We conducted an empirical study (N = 248) of LLM agents as daily assistants in six commonly occurring tasks with different levels of risk typically associated with them (e.g., flight ticket booking and credit card payments). To ensure user agency and control over the LLM agent, we adopted LLM agents in a plan-then-execute manner, wherein the agents conducted step-wise planning and step-by-step execution in a simulation environment. We analyzed how user involvement at each stage affects their trust and collaborative team performance. Our findings demonstrate that LLM agents can be a double-edged sword -- (1) they can work well when a high-quality plan and necessary user involvement in execution are available, and (2) users can easily mistrust the LLM agents with plans that seem plausible. We synthesized key insights for using LLM agents as daily assistants to calibrate user trust and achieve better overall task outcomes. Our work has important implications for the future design of daily assistants and human-AI collaboration with LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01387v1">TeLL-Drive: Enhancing Autonomous Driving with Teacher LLM-Guided Deep Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Although Deep Reinforcement Learning (DRL) and Large Language Models (LLMs) each show promise in addressing decision-making challenges in autonomous driving, DRL often suffers from high sample complexity, while LLMs have difficulty ensuring real-time decision making. To address these limitations, we propose TeLL-Drive, a hybrid framework that integrates an Teacher LLM to guide an attention-based Student DRL policy. By incorporating risk metrics, historical scenario retrieval, and domain heuristics into context-rich prompts, the LLM produces high-level driving strategies through chain-of-thought reasoning. A self-attention mechanism then fuses these strategies with the DRL agent's exploration, accelerating policy convergence and boosting robustness across diverse driving conditions. Our experimental results, evaluated across multiple traffic scenarios, show that TeLL-Drive outperforms existing baseline methods, including other LLM-based approaches, in terms of success rates, average returns, and real-time feasibility. Ablation studies underscore the importance of each model component, especially the synergy between the attention mechanism and LLM-driven guidance. These findings suggest that TeLL-Drive significantly enhances both the adaptability and safety of autonomous driving systems, while offering a more efficient and scalable approach for policy learning. Full validation results are available on our website.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01349v1">Bias Beware: The Impact of Cognitive Biases on LLM-Driven Product Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized product recommendation systems, yet their susceptibility to adversarial manipulation poses critical challenges, particularly in real-world commercial applications. Our approach is the first one to tap into human psychological principles, seamlessly modifying product descriptions, making these adversarial manipulations hard to detect. In this work, we investigate cognitive biases as black-box adversarial strategies, drawing parallels between their effects on LLMs and human purchasing behavior. Through extensive experiments on LLMs of varying scales, we reveal significant vulnerabilities in their use as recommenders, providing critical insights into safeguarding these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01705v1">Progressive Binarization with Semi-Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, but their high computational and memory demands pose challenges for deployment on resource-constrained devices. Binarization, as an efficient compression method that reduces model weights to just 1 bit, significantly lowers both computational and memory requirements. Despite this, the binarized LLM still contains redundancy, which can be further compressed. Semi-structured pruning provides a promising approach to achieve this, which offers a better trade-off between model performance and hardware efficiency. However, simply combining binarization with semi-structured pruning can lead to a significant performance drop. To address this issue, we propose a Progressive Binarization with Semi-Structured Pruning (PBS$^2$P) method for LLM compression. We first propose a Stepwise semi-structured Pruning with Binarization Optimization (SPBO). Our optimization strategy significantly reduces the total error caused by pruning and binarization, even below that of the no-pruning scenario. Furthermore, we design a Coarse-to-Fine Search (CFS) method to select pruning elements more effectively. Extensive experiments demonstrate that PBS$^2$P achieves superior accuracy across various LLM families and evaluation metrics, noticeably outperforming state-of-the-art (SOTA) binary PTQ methods. The code and models will be available at https://github.com/XIANGLONGYAN/PBS2P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01298v1">Augmented Knowledge Graph Querying leveraging LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-03
    </div>
    <details class="paper-abstract">
      Adopting Knowledge Graphs (KGs) as a structured, semantic-oriented, data representation model has significantly improved data integration, reasoning, and querying capabilities across different domains. This is especially true in modern scenarios such as Industry 5.0, in which the integration of data produced by humans, smart devices, and production processes plays a crucial role. However, the management, retrieval, and visualization of data from a KG using formal query languages can be difficult for non-expert users due to their technical complexity, thus limiting their usage inside industrial environments. For this reason, we introduce SparqLLM, a framework that utilizes a Retrieval-Augmented Generation (RAG) solution, to enhance the querying of Knowledge Graphs (KGs). SparqLLM executes the Extract, Transform, and Load (ETL) pipeline to construct KGs from raw data. It also features a natural language interface powered by Large Language Models (LLMs) to enable automatic SPARQL query generation. By integrating template-based methods as retrieved-context for the LLM, SparqLLM enhances query reliability and reduces semantic errors, ensuring more accurate and efficient KG interactions. Moreover, to improve usability, the system incorporates a dynamic visualization dashboard that adapts to the structure of the retrieved data, presenting the query results in an intuitive format. Rigorous experimental evaluations demonstrate that SparqLLM achieves high query accuracy, improved robustness, and user-friendly interaction with KGs, establishing it as a scalable solution to access semantic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01273v1">Analysis of Student-LLM Interaction in a Software Engineering Project</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly competent across various domains, educators are showing a growing interest in integrating these LLMs into the learning process. Especially in software engineering, LLMs have demonstrated qualitatively better capabilities in code summarization, code generation, and debugging. Despite various research on LLMs for software engineering tasks in practice, limited research captures the benefits of LLMs for pedagogical advancements and their impact on the student learning process. To this extent, we analyze 126 undergraduate students' interaction with an AI assistant during a 13-week semester to understand the benefits of AI for software engineering learning. We analyze the conversations, code generated, code utilized, and the human intervention levels to integrate the code into the code base. Our findings suggest that students prefer ChatGPT over CoPilot. Our analysis also finds that ChatGPT generates responses with lower computational complexity compared to CoPilot. Furthermore, conversational-based interaction helps improve the quality of the code generated compared to auto-generated code. Early adoption of LLMs in software engineering is crucial to remain competitive in the rapidly developing landscape. Hence, the next generation of software engineers must acquire the necessary skills to interact with AI to improve productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01205v1">OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches</a></div>
    <div class="paper-meta">
      📅 2025-02-03
      | 💬 To be published in RESOURCEFUL 2025
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) systems often introduce errors when transcribing historical documents, leaving room for post-correction to improve text quality. This study evaluates the use of open-weight LLMs for OCR error correction in historical English and Finnish datasets. We explore various strategies, including parameter optimization, quantization, segment length effects, and text continuation methods. Our results demonstrate that while modern LLMs show promise in reducing character error rates (CER) in English, a practically useful performance for Finnish was not reached. Our findings highlight the potential and limitations of LLMs in scaling OCR post-correction for large historical corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14931v2">Do LLMs Dream of Ontologies?</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across diverse natural language processing tasks, yet their ability to memorize structured knowledge remains underexplored. In this paper, we investigate the extent to which general-purpose pre-trained LLMs retain and correctly reproduce concept identifier (ID)-label associations from publicly available ontologies. We conduct a systematic evaluation across multiple ontological resources, including the Gene Ontology, Uberon, Wikidata, and ICD-10, using LLMs such as Pythia-12B, Gemini-1.5-Flash, GPT-3.5, and GPT-4. Our findings reveal that only a small fraction of ontological concepts is accurately memorized, with GPT-4 demonstrating the highest performance. To understand why certain concepts are memorized more effectively than others, we analyze the relationship between memorization accuracy and concept popularity on the Web. Our results indicate a strong correlation between the frequency of a concept's occurrence online and the likelihood of accurately retrieving its ID from the label. This suggests that LLMs primarily acquire such knowledge through indirect textual exposure rather than directly from structured ontological resources. Furthermore, we introduce new metrics to quantify prediction invariance, demonstrating that the stability of model responses across variations in prompt language and temperature settings can serve as a proxy for estimating memorization robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07927v2">Gandalf the Red: Adaptive Security for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally
    </div>
    <details class="paper-abstract">
      Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14427v2">GraphSOS: Graph Sampling and Order Selection to Help LLMs Understand Graphs Better</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The success of Large Language Models (LLMs) in various domains has led researchers to apply them to graph-related problems by converting graph data into natural language text. However, unlike graph data, natural language inherently has sequential order. We observe a counter-intuitive fact that when the order of nodes or edges in the natural language description of a graph is shuffled, despite describing the same graph, model performance fluctuates between high performance and random guessing. Additionally, due to LLMs' limited input context length, current methods typically randomly sample neighbors of target nodes as representatives of their neighborhood, which may not always be effective for accurate reasoning. To address these gaps, we introduce GraphSOS (Graph Sampling and Order Selection). This novel model framework features an Order Selector Module to ensure proper serialization order of the graph and a Subgraph Sampling Module to sample subgraphs with better structure for better reasoning. Furthermore, we propose Graph CoT obtained through distillation, and enhance LLM's reasoning and zero-shot learning capabilities for graph tasks through instruction tuning. Experiments on multiple datasets for node classification and graph question-answering demonstrate that GraphSOS improves LLMs' performance and generalization ability on graph tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16383v2">RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Key-Value (KV) cache facilitates efficient large language models (LLMs) inference by avoiding recomputation of past KVs. As the batch size and context length increase, the oversized KV caches become a significant memory bottleneck, highlighting the need for efficient compression. Existing KV quantization rely on fine-grained quantization or the retention of a significant portion of high bit-widths caches, both of which compromise compression ratio and often fail to maintain robustness at extremely low average bit-widths. In this work, we explore the potential of rotation technique for 2-bit KV quantization and propose RotateKV, which achieves accurate and robust performance through the following innovations: (i) Outlier-Aware Rotation, which utilizes channel-reordering to adapt the rotations to varying channel-wise outlier distributions without sacrificing the computational efficiency of the fast Walsh-Hadamard transform (FWHT); (ii) Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position embedding (RoPE) on proposed outlier-aware rotation and further smooths outliers across heads; (iii) Attention-Sink-Aware Quantization, which leverages the massive activations to precisely identify and protect attention sinks. RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning and long-context capabilities, with less than 1.7\% degradation on GSM8K, outperforming existing methods even at lower average bit-widths. RotateKV also showcases a 3.97x reduction in peak memory usage, supports 5.75x larger batch sizes, and achieves a 2.32x speedup in decoding stage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07288v2">LLM-Net: Democratizing LLMs-as-a-Service through Blockchain-based Expert Networks</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The centralization of Large Language Models (LLMs) development has created significant barriers to AI advancement, limiting the democratization of these powerful technologies. This centralization, coupled with the scarcity of high-quality training data and mounting complexity of maintaining comprehensive expertise across rapidly expanding knowledge domains, poses critical challenges to the continued growth of LLMs. While solutions like Retrieval-Augmented Generation (RAG) offer potential remedies, maintaining up-to-date expert knowledge across diverse domains remains a significant challenge, particularly given the exponential growth of specialized information. This paper introduces LLMs Networks (LLM-Net), a blockchain-based framework that democratizes LLMs-as-a-Service through a decentralized network of specialized LLM providers. By leveraging collective computational resources and distributed domain expertise, LLM-Net incorporates fine-tuned expert models for various specific domains, ensuring sustained knowledge growth while maintaining service quality through collaborative prompting mechanisms. The framework's robust design includes blockchain technology for transparent transaction and performance validation, establishing an immutable record of service delivery. Our simulation, built on top of state-of-the-art LLMs such as Claude 3.5 Sonnet, Llama 3.1, Grok-2, and GPT-4o, validates the effectiveness of the reputation-based mechanism in maintaining service quality by selecting high-performing respondents (LLM providers). Thereby it demonstrates the potential of LLM-Net to sustain AI advancement through the integration of decentralized expertise and blockchain-based accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00922v1">Huff-LLM: End-to-End Lossless Compression for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      As they become more capable, large language models (LLMs) have continued to rapidly increase in size. This has exacerbated the difficulty in running state of the art LLMs on small, edge devices. Standard techniques advocate solving this problem through lossy compression techniques such as quantization or pruning. However, such compression techniques are lossy, and have been shown to change model behavior in unpredictable manners. We propose Huff-LLM, an \emph{end-to-end, lossless} model compression method that lets users store LLM weights in compressed format \emph{everywhere} -- cloud, disk, main memory, and even in on-chip memory/buffers. This allows us to not only load larger models in main memory, but also reduces bandwidth required to load weights on chip, and makes more efficient use of on-chip weight buffers. In addition to the memory savings achieved via compression, we also show latency and energy efficiency improvements when performing inference with the compressed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00916v1">The Accuracy, Robustness, and Readability of LLM-Generated Sustainability-Related Word Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 NLP4Ecology Workshop 2025
    </div>
    <details class="paper-abstract">
      A common language with standardized definitions is crucial for effective climate discussions. However, concerns exist about LLMs misrepresenting climate terms. We compared 300 official IPCC glossary definitions with those generated by GPT-4o-mini, Llama3.1 8B, and Mistral 7B, analyzing adherence, robustness, and readability using SBERT sentence embeddings. The LLMs scored an average adherence of $0.57-0.59 \pm 0.15$, and their definitions proved harder to read than the originals. Model-generated definitions vary mainly among words with multiple or ambiguous definitions, showing the potential to highlight terms that need standardization. The results show how LLMs could support environmental discourse while emphasizing the need to align model outputs with established terminology for clarity and consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00899v1">HASSLE-free: A unified Framework for Sparse plus Low-Rank Matrix Decomposition for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The impressive capabilities of large foundation models come at a cost of substantial computing resources to serve them. Compressing these pre-trained models is of practical interest as it can democratize deploying them to the machine learning community at large by lowering the costs associated with inference. A promising compression scheme is to decompose foundation models' dense weights into a sum of sparse plus low-rank matrices. In this paper, we design a unified framework coined HASSLE-free for (semi-structured) sparse plus low-rank matrix decomposition of foundation models. Our framework introduces the local layer-wise reconstruction error objective for this decomposition, we demonstrate that prior work solves a relaxation of this optimization problem; and we provide efficient and scalable methods to minimize the exact introduced optimization problem. HASSLE-free substantially outperforms state-of-the-art methods in terms of the introduced objective and a wide range of LLM evaluation benchmarks. For the Llama3-8B model with a 2:4 sparsity component plus a 64-rank component decomposition, a compression scheme for which recent work shows important inference acceleration on GPUs, HASSLE-free reduces the test perplexity by 12% for the WikiText-2 dataset and reduces the gap (compared to the dense model) of the average of eight popular zero-shot tasks by 15% compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00894v1">MorphBPE: A Morpho-Aware Tokenizer Bridging Linguistic Complexity for Efficient LLM Training Across Morphologies</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Tokenization is fundamental to Natural Language Processing (NLP), directly impacting model efficiency and linguistic fidelity. While Byte Pair Encoding (BPE) is widely used in Large Language Models (LLMs), it often disregards morpheme boundaries, leading to suboptimal segmentation, particularly in morphologically rich languages. We introduce MorphBPE, a morphology-aware extension of BPE that integrates linguistic structure into subword tokenization while preserving statistical efficiency. Additionally, we propose two morphology-based evaluation metrics: (i) Morphological Consistency F1-Score, which quantifies the consistency between morpheme sharing and token sharing, contributing to LLM training convergence, and (ii) Morphological Edit Distance, which measures alignment between morphemes and tokens concerning interpretability. Experiments on English, Russian, Hungarian, and Arabic across 300M and 1B parameter LLMs demonstrate that MorphBPE consistently reduces cross-entropy loss, accelerates convergence, and improves morphological alignment scores. Fully compatible with existing LLM pipelines, MorphBPE requires minimal modifications for integration. The MorphBPE codebase and tokenizer playground will be available at: https://github.com/llm-lab-org/MorphBPE and https://tokenizer.llm-lab.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00840v1">Activation Approximations Can Incur Safety Vulnerabilities Even in Aligned LLMs: Comprehensive Analysis and Defense</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have showcased remarkable capabilities across various domains. Accompanying the evolving capabilities and expanding deployment scenarios of LLMs, their deployment challenges escalate due to their sheer scale and the advanced yet complex activation designs prevalent in notable model series, such as Llama, Gemma, and Mistral. These challenges have become particularly pronounced in resource-constrained deployment scenarios, where mitigating inference efficiency bottlenecks is imperative. Among various recent efforts, activation approximation has emerged as a promising avenue for pursuing inference efficiency, sometimes considered indispensable in applications such as private inference. Despite achieving substantial speedups with minimal impact on utility, even appearing sound and practical for real-world deployment, the safety implications of activation approximations remain unclear. In this work, we fill this critical gap in LLM safety by conducting the first systematic safety evaluation of activation approximations. Our safety vetting spans seven sota techniques across three popular categories, revealing consistent safety degradation across ten safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00829v1">A Comprehensive Analysis on LLM-based Node Classification Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Node classification is a fundamental task in graph analysis, with broad applications across various fields. Recent breakthroughs in Large Language Models (LLMs) have enabled LLM-based approaches for this task. Although many studies demonstrate the impressive performance of LLM-based methods, the lack of clear design guidelines may hinder their practical application. In this work, we aim to establish such guidelines through a fair and systematic comparison of these algorithms. As a first step, we developed LLMNodeBed, a comprehensive codebase and testbed for node classification using LLMs. It includes ten datasets, eight LLM-based algorithms, and three learning paradigms, and is designed for easy extension with new methods and datasets. Subsequently, we conducted extensive experiments, training and evaluating over 2,200 models, to determine the key settings (e.g., learning paradigms and homophily) and components (e.g., model size) that affect performance. Our findings uncover eight insights, e.g., (1) LLM-based methods can significantly outperform traditional methods in a semi-supervised setting, while the advantage is marginal in a supervised setting; (2) Graph Foundation Models can beat open-source LLMs but still fall short of strong LLMs like GPT-4o in a zero-shot setting. We hope that the release of LLMNodeBed, along with our insights, will facilitate reproducible research and inspire future studies in this field. Codes and datasets are released at \href{https://llmnodebed.github.io/}{https://llmnodebed.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00808v1">Synthetic Artifact Auditing: Tracing LLM-Generated Synthetic Data Usage in Downstream Applications</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 To Appear in the 34th USENIX Security Symposium, August 13-15, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated the generation of high-quality, cost-effective synthetic data for developing downstream models and conducting statistical analyses in various domains. However, the increased reliance on synthetic data may pose potential negative impacts. Numerous studies have demonstrated that LLM-generated synthetic data can perpetuate and even amplify societal biases and stereotypes, and produce erroneous outputs known as ``hallucinations'' that deviate from factual knowledge. In this paper, we aim to audit artifacts, such as classifiers, generators, or statistical plots, to identify those trained on or derived from synthetic data and raise user awareness, thereby reducing unexpected consequences and risks in downstream applications. To this end, we take the first step to introduce synthetic artifact auditing to assess whether a given artifact is derived from LLM-generated synthetic data. We then propose an auditing framework with three methods including metric-based auditing, tuning-based auditing, and classification-based auditing. These methods operate without requiring the artifact owner to disclose proprietary training details. We evaluate our auditing framework on three text classification tasks, two text summarization tasks, and two data visualization tasks across three training scenarios. Our evaluation demonstrates the effectiveness of all proposed auditing methods across all these tasks. For instance, black-box metric-based auditing can achieve an average accuracy of $0.868 \pm 0.071$ for auditing classifiers and $0.880 \pm 0.052$ for auditing generators using only 200 random queries across three scenarios. We hope our research will enhance model transparency and regulatory compliance, ensuring the ethical and responsible use of synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00792v1">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Accepted by WWW 2025
    </div>
    <details class="paper-abstract">
      Real-Time Bidding (RTB) enables advertisers to place competitive bids on impression opportunities instantaneously, striving for cost-effectiveness in a highly competitive landscape. Although RTB has widely benefited from the utilization of technologies such as deep learning and reinforcement learning, the reliability of related methods often encounters challenges due to the discrepancies between online and offline environments and the rapid fluctuations of online bidding. To handle these challenges, RTBAgent is proposed as the first RTB agent system based on large language models (LLMs), which synchronizes real competitive advertising bidding environments and obtains bidding prices through an integrated decision-making process. Specifically, obtaining reasoning ability through LLMs, RTBAgent is further tailored to be more professional for RTB via involved auxiliary modules, i.e., click-through rate estimation model, expert strategy knowledge, and daily reflection. In addition, we propose a two-step decision-making process and multi-memory retrieval mechanism, which enables RTBAgent to review historical decisions and transaction records and subsequently make decisions more adaptive to market changes in real-time bidding. Empirical testing with real advertising datasets demonstrates that RTBAgent significantly enhances profitability. The RTBAgent code will be publicly accessible at: https://github.com/CaiLeng/RTBAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00735v1">From Compliance to Exploitation: Jailbreak Prompt Attacks on Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the frontier multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. To better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flank Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios. These findings highlight both the potency of prompt-based obfuscation in voice-enabled contexts and the limitations of current LLMs' moderation safeguards and the urgent need for advanced defense strategies to address the challenges posed by evolving, context-rich attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00728v1">Meta-Prompt Optimization for LLM-Based Sequential Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been employed as agents to solve sequential decision-making tasks such as Bayesian optimization and multi-armed bandits (MAB). These works usually adopt an LLM for sequential action selection by providing it with a fixed, manually designed meta-prompt. However, numerous previous works have found that the prompt has a significant impact on the performance of the LLM, which calls for a method to automatically optimize the meta-prompt for LLM-based agents. Unfortunately, the non-stationarity in the reward observations during LLM-based sequential decision-making makes meta-prompt optimization highly challenging. To address this challenge, we draw inspirations from adversarial bandit algorithms, which are inherently capable of handling non-stationary reward observations. Building on this foundation, we propose our EXPonential-weight algorithm for prompt Optimization} (EXPO) to automatically optimize the task description and meta-instruction in the meta-prompt for LLM-based agents. We also extend EXPO to additionally optimize the exemplars (i.e., history of interactions) in the meta-prompt to further enhance the performance, hence introducing our EXPO-ES algorithm. We use extensive experiments to show that our algorithms significantly improve the performance of LLM-based sequential decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00722v1">Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to increasingly diverse requests, accompanied with varying resource (compute and memory) demands to serve them. However, this in turn degrades the cost-efficiency of LLM serving as common practices primarily rely on homogeneous GPU resources. In response to this problem, this work conducts a thorough study about serving LLMs over heterogeneous GPU resources on cloud platforms. The rationale is that different GPU types exhibit distinct compute and memory characteristics, aligning well with the divergent resource demands of diverse requests. Particularly, through comprehensive benchmarking, we discover that the cost-efficiency of LLM serving can be substantially optimized by meticulously determining GPU composition, deployment configurations, and workload assignments. Subsequently, we design a scheduling algorithm via mixed-integer linear programming, aiming at deducing the most cost-efficient serving plan under the constraints of price budget and real-time GPU availability. Remarkably, our approach effectively outperforms homogeneous and heterogeneous baselines under a wide array of scenarios, covering diverse workload traces, varying GPU availablilities, and multi-model serving. This casts new light on more accessible and efficient LLM serving over heterogeneous cloud resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01683v1">LLM-Powered Benchmark Factory: Reliable, Generic, and Efficient</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has led to a surge in both model supply and application demands. To facilitate effective matching between them, reliable, generic and efficient benchmark generators are widely needed. However, human annotators are constrained by inefficiency, and current LLM benchmark generators not only lack generalizability but also struggle with limited reliability, as they lack a comprehensive evaluation framework for validation and optimization. To fill this gap, we first propose an automated and unbiased evaluation framework, structured around four dimensions and ten criteria. Under this framework, we carefully analyze the advantages and weaknesses of directly prompting LLMs as generic benchmark generators. To enhance the reliability, we introduce a series of methods to address the identified weaknesses and integrate them as BenchMaker. Experiments across multiple LLMs and tasks confirm that BenchMaker achieves superior or comparable performance to human-annotated benchmarks on all metrics, highlighting its generalizability and reliability. More importantly, it delivers highly consistent evaluation results across 12 LLMs (0.967 Pearson correlation against MMLU-Pro), while taking only $0.005 and 0.38 minutes per sample.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00689v1">Leveraging LLMs for Dynamic IoT Systems Generation through Mixed-Initiative Interaction</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      IoT systems face significant challenges in adapting to user needs, which are often under-specified and evolve with changing environmental contexts. To address these complexities, users should be able to explore possibilities, while IoT systems must learn and support users in the process of providing proper services, e.g., to serve novel experiences. The IoT-Together paradigm aims to meet this demand through the Mixed-Initiative Interaction (MII) paradigm that facilitates a collaborative synergy between users and IoT systems, enabling the co-creation of intelligent and adaptive solutions that are precisely aligned with user-defined goals. This work advances IoT-Together by integrating Large Language Models (LLMs) into its architecture. Our approach enables intelligent goal interpretation through a multi-pass dialogue framework and dynamic service generation at runtime according to user needs. To demonstrate the efficacy of our methodology, we design and implement the system in the context of a smart city tourism case study. We evaluate the system's performance using agent-based simulation and user studies. Results indicate efficient and accurate service identification and high adaptation quality. The empirical evidence indicates that the integration of Large Language Models (LLMs) into IoT architectures can significantly enhance the architectural adaptability of the system while ensuring real-world usability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00677v1">LLM-based event log analysis techniques: A survey</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Event log analysis is an important task that security professionals undertake. Event logs record key information on activities that occur on computing devices, and due to the substantial number of events generated, they consume a large amount of time and resources to analyse. This demanding and repetitive task is also prone to errors. To address these concerns, researchers have developed automated techniques to improve the event log analysis process. Large Language Models (LLMs) have recently demonstrated the ability to successfully perform a wide range of tasks that individuals would usually partake in, to high standards, and at a pace and degree of complexity that outperform humans. Due to this, researchers are rapidly investigating the use of LLMs for event log analysis. This includes fine-tuning, Retrieval-Augmented Generation (RAG) and in-context learning, which affect performance. These works demonstrate good progress, yet there is a need to understand the developing body of knowledge, identify commonalities between works, and identify key challenges and potential solutions to further developments in this domain. This paper aims to survey LLM-based event log analysis techniques, providing readers with an in-depth overview of the domain, gaps identified in previous research, and concluding with potential avenues to explore in future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00657v1">LLM Safety Alignment is Divergence Estimation in Disguise</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      We propose a theoretical framework demonstrating that popular Large Language Model (LLM) alignment methods, including Reinforcement Learning from Human Feedback (RLHF) and alternatives, fundamentally function as divergence estimators between aligned (preferred or safe) and unaligned (less-preferred or harmful) distributions. This explains the separation phenomenon between safe and harmful prompts in the model hidden representation after alignment. Inspired by the theoretical results, we identify that some alignment methods are better than others in terms of separation and, introduce a new method, KLDO, and further demonstrate the implication of our theories. We advocate for compliance-refusal datasets over preference datasets to enhance safety alignment, supported by both theoretical reasoning and empirical evidence. Additionally, to quantify safety separation, we leverage a distance metric in the representation space and statistically validate its efficacy as a statistical significant indicator of LLM resilience against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00602v1">Mitigating Heterogeneous Token Overfitting in LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on various natural language tasks. However, they are trained on static corpora and their knowledge can become outdated quickly in the fast-changing world. This motivates the development of knowledge editing (KE) to update specific knowledge in LLMs without changing unrelated others or compromising their pre-trained capabilities. Previous efforts sought to update a small amount of parameters of a LLM and proved effective for making selective updates. Nonetheless, the edited LLM often exhibits degraded ability to reason about the new knowledge. In this work, we identify a key issue: heterogeneous token overfitting (HTO), where the LLM overfits different tokens in the provided knowledge at varying rates. To tackle this, we propose OVERTONE, a token-level smoothing method that mitigates HTO by adaptively refining the target distribution. Theoretically, OVERTONE offers better parameter updates with negligible computation overhead. It also induces an implicit DPO but does not require preference data pairs. Extensive experiments across four editing methods, two LLMs, and diverse scenarios demonstrate the effectiveness and versatility of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02626v3">Time-Reversal Provides Unsupervised Feedback to LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 Accepted as a spotlight in NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are typically trained to predict in the forward direction of time. However, recent works have shown that prompting these models to look back and critique their own generations can produce useful feedback. Motivated by this, we explore the question of whether LLMs can be empowered to think (predict and score) backwards to provide unsupervised feedback that complements forward LLMs. Towards this, we introduce Time Reversed Language Models (TRLMs), which can score and generate queries when conditioned on responses, effectively functioning in the reverse direction of time. Further, to effectively infer in the response to query direction, we pre-train and fine-tune a language model (TRLM-Ba) in the reverse token order from scratch. We show empirically (and theoretically in a stylized setting) that time-reversed models can indeed complement forward model predictions when used to score the query given response for re-ranking multiple forward generations. We obtain up to 5\% improvement on the widely used AlpacaEval Leaderboard over the competent baseline of best-of-N re-ranking using self log-perplexity scores. We further show that TRLM scoring outperforms conventional forward scoring of response given query, resulting in significant gains in applications such as citation generation and passage retrieval. We next leverage the generative ability of TRLM to augment or provide unsupervised feedback to input safety filters of LLMs, demonstrating a drastic reduction in false negative rate with negligible impact on false positive rates against several attacks published on the popular JailbreakBench leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12004v2">The Open Source Advantage in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-02-02
      | 💬 9 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly advanced natural language processing, driving significant breakthroughs in tasks such as text generation, machine translation, and domain-specific reasoning. The field now faces a critical dilemma in its approach: closed-source models like GPT-4 deliver state-of-the-art performance but restrict reproducibility, accessibility, and external oversight, while open-source frameworks like LLaMA and Mixtral democratize access, foster collaboration, and support diverse applications, achieving competitive results through techniques like instruction tuning and LoRA. Hybrid approaches address challenges like bias mitigation and resource accessibility by combining the scalability of closed-source systems with the transparency and inclusivity of open-source framework. However, in this position paper, we argue that open-source remains the most robust path for advancing LLM research and ethical deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05541v3">Customizable LLM-Powered Chatbot for Behavioral Science Research</a></div>
    <div class="paper-meta">
      📅 2025-02-02
    </div>
    <details class="paper-abstract">
      The rapid advancement of Artificial Intelligence has resulted in the advent of Large Language Models (LLMs) with the capacity to produce text that closely resembles human communication. These models have been seamlessly integrated into diverse applications, enabling interactive and responsive communication across multiple platforms. The potential utility of chatbots transcends these traditional applications, particularly in research contexts, wherein they can offer valuable insights and facilitate the design of innovative experiments. In this study, we present a Customizable LLM-Powered Chatbot (CLPC), a web-based chatbot system designed to assist in behavioral science research. The system is meticulously designed to function as an experimental instrument rather than a conventional chatbot, necessitating users to input a username and experiment code upon access. This setup facilitates precise data cross-referencing, thereby augmenting the integrity and applicability of the data collected for research purposes. It can be easily expanded to accommodate new basic events as needed; and it allows researchers to integrate their own logging events without the necessity of implementing a separate logging mechanism. It is worth noting that our system was built to assist primarily behavioral science research but is not limited to it, it can easily be adapted to assist information retrieval research or interacting with chat bot agents in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14808v2">HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated a wide range of applications with distinct service-level objectives (SLOs), from latency-sensitive online tasks like interactive chatbots to throughput-oriented offline workloads like document summarization. The existing deployment model, which dedicates machines to each workload, simplifies SLO management but often leads to poor resource utilization. This paper introduces HyGen, an interference-aware LLM serving system that enables efficient co-location of online and offline workloads while preserving latency requirements. HyGen incorporates two key innovations: (1) performance control mechanisms, including a latency predictor to estimate batch execution time and an SLO-aware profiler to quantify latency interference, and (2) SLO-aware offline scheduling policies that maximize serving throughput and prevent starvation, without compromising online serving latency. Our evaluation on production workloads shows that HyGen achieves up to 3.87x overall throughput and 5.84x offline throughput gains over online and hybrid serving baselines, respectively, while strictly satisfying latency SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13865v2">Breaking Information Cocoons: A Hyperbolic Graph-LLM Framework for Exploration and Exploitation in Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Modern recommender systems often create information cocoons, restricting users' exposure to diverse content. A key challenge lies in balancing content exploration and exploitation while allowing users to adjust their recommendation preferences. Intuitively, this balance can be modeled as a tree-structured representation, where depth search facilitates exploitation and breadth search enables exploration. However, existing approaches face two fundamental limitations: Euclidean methods struggle to capture hierarchical structures, while hyperbolic methods, despite their superior hierarchical modeling, lack semantic understanding of user and item profiles and fail to provide a principled mechanism for balancing exploration and exploitation. To address these challenges, we propose HERec, a hyperbolic graph-LLM framework that effectively balances exploration and exploitation in recommender systems. Our framework introduces two key innovations: (1) a hierarchical-aware graph-LLM mechanism that jointly aligns textual descriptions with user-item collaborative information in hyperbolic space, and (2) a hierarchical representation structure that enables user-adjustable exploration-exploitation trade-offs. Extensive experiments demonstrate that HERec consistently outperforms both Euclidean and hyperbolic baselines, achieving up to 5.49% improvement in utility metrics and 11.39% increase in diversity metrics, effectively mitigating information cocoons. We open-source our model implementation at https://github.com/Martin-qyma/HERec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17183v2">LLM Evaluation Based on Aerospace Manufacturing Expertise: Automated Generation and Multi-Model Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 conference paper
    </div>
    <details class="paper-abstract">
      Aerospace manufacturing demands exceptionally high precision in technical parameters. The remarkable performance of Large Language Models (LLMs), such as GPT-4 and QWen, in Natural Language Processing has sparked industry interest in their application to tasks including process design, material selection, and tool information retrieval. However, LLMs are prone to generating "hallucinations" in specialized domains, producing inaccurate or false information that poses significant risks to the quality of aerospace products and flight safety. This paper introduces a set of evaluation metrics tailored for LLMs in aerospace manufacturing, aiming to assess their accuracy by analyzing their performance in answering questions grounded in professional knowledge. Firstly, key information is extracted through in-depth textual analysis of classic aerospace manufacturing textbooks and guidelines. Subsequently, utilizing LLM generation techniques, we meticulously construct multiple-choice questions with multiple correct answers of varying difficulty. Following this, different LLM models are employed to answer these questions, and their accuracy is recorded. Experimental results demonstrate that the capabilities of LLMs in aerospace professional knowledge are in urgent need of improvement. This study provides a theoretical foundation and practical guidance for the application of LLMs in aerospace manufacturing, addressing a critical gap in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v4">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Corrected typos & more discussion on reasoning models 33 pages, 9 figures. arXiv admin note: text overlap with arXiv:2310.05470 by other authors
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17626v2">Tracking the Feature Dynamics in LLM Training: A Mechanistic Study</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we: (1) introduce SAE-Track, a novel method to efficiently obtain a continual series of SAEs; (2) mechanistically investigate feature formation and develop a progress measure for it ; and (3) analyze and visualize feature drift during training. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13382v3">VTG-LLM: Integrating Timestamp Knowledge into Video LLMs for Enhanced Video Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 AAAI 2025
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG) strives to accurately pinpoint event timestamps in a specific video using linguistic queries, significantly impacting downstream tasks like video browsing and editing. Unlike traditional task-specific models, Video Large Language Models (video LLMs) can handle multiple tasks concurrently in a zero-shot manner. Consequently, exploring the application of video LLMs for VTG tasks has become a burgeoning research area. However, despite considerable advancements in video content understanding, video LLMs often struggle to accurately pinpoint timestamps within videos, limiting their effectiveness in VTG tasks. To address this, we introduce VTG-LLM, a model designed to enhance video LLMs' timestamp localization abilities. Our approach includes: (1) effectively integrating timestamp knowledge into visual tokens; (2) incorporating absolute-time tokens to manage timestamp knowledge without concept shifts; and (3) introducing a lightweight, high-performance, slot-based token compression technique designed to accommodate the demands of a large number of frames to be sampled for VTG tasks. Additionally, we present VTG-IT-120K, a collection of publicly available VTG datasets that we have re-annotated to improve upon low-quality annotations. Our comprehensive experiments demonstrate the superior performance of VTG-LLM in comparison to other video LLM methods across a variety of VTG tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03993v2">Assessing LLMs for Zero-shot Abstractive Summarization Through the Lens of Relevance Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Accepted to NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved state-of-the-art performance at zero-shot generation of abstractive summaries for given articles. However, little is known about the robustness of such a process of zero-shot summarization. To bridge this gap, we propose relevance paraphrasing, a simple strategy that can be used to measure the robustness of LLMs as summarizers. The relevance paraphrasing approach identifies the most relevant sentences that contribute to generating an ideal summary, and then paraphrases these inputs to obtain a minimally perturbed dataset. Then, by evaluating model performance for summarization on both the original and perturbed datasets, we can assess the LLM's one aspect of robustness. We conduct extensive experiments with relevance paraphrasing on 4 diverse datasets, as well as 4 LLMs of different sizes (GPT-3.5-Turbo, Llama-2-13B, Mistral-7B, and Dolly-v2-7B). Our results indicate that LLMs are not consistent summarizers for the minimally perturbed articles, necessitating further improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v2">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00511v1">Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated remarkable reasoning capabilities. However, single-shot inference often yields unreliable results for complex reasoning tasks, leading researchers to explore multiple reasoning paths through methods such as perplexity and self-consistency. In this paper, we present the first theoretical error decomposition analysis of these techniques, breaking down their error into estimation error and model error. Our analysis reveals a fundamental trade-off: perplexity methods suffer from substantial model error due to the absence of a proper consistency function, while self-consistency exhibits high estimation error due to a slow error convergence rate. To overcome these limitations, we propose Reasoning-Pruning Perplexity Consistency (RPC). This approach combines Perplexity Consistency, which seamlessly integrates LLM perplexity with self-consistency, and Reasoning Pruning, which eliminates low-probability reasoning paths to effectively prevent the degeneration of estimation error reduction. Theoretical analysis demonstrates that RPC not only accelerates the convergence rate of estimation error to an exponential level but also holds strong potential for further reducing model error. Extensive empirical evaluations on seven benchmark datasets confirm that RPC can significantly improve reasoning performance, sample efficiency, and confidence reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00510v1">Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents frameworks often employ modular architectures, incorporating components such as planning, reasoning, action execution, and reflection to tackle complex tasks. However, quantifying the contribution of each module to overall system performance remains a significant challenge, impeding optimization and interpretability. To address this, we introduce CapaBench (Capability-level Assessment Benchmark), an evaluation framework grounded in cooperative game theory's Shapley Value, which systematically measures the marginal impact of individual modules and their interactions within an agent's architecture. By replacing default modules with test variants across all possible combinations, CapaBench provides a principle method for attributing performance contributions. Key contributions include: (1) We are the first to propose a Shapley Value-based methodology for quantifying the contributions of capabilities in LLM agents; (2) Modules with high Shapley Values consistently lead to predictable performance gains when combined, enabling targeted optimization; and (3) We build a multi-round dataset of over 1,000 entries spanning diverse domains and practical task scenarios, enabling comprehensive evaluation of agent capabilities. CapaBench bridges the gap between component-level evaluation and holistic system assessment, providing actionable insights for optimizing modular LLM agents and advancing their deployment in complex, real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00439v1">UniAttn: Reducing Inference Costs via Softmax Unification for Post-Training LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 11 pages, 4 figures. Preprint, under review
    </div>
    <details class="paper-abstract">
      Post-training is essential for adapting Large Language Models (LLMs) to real-world applications. Deploying post-trained models faces significant challenges due to substantial memory overhead and noticeable inference latency. Existing work has identified significant redundancies in LLMs and proposed efficient architectures, namely intra-layer KV sharing and cross-layer KV sharing. However, intra-layer KV sharing still results in high inference costs, while cross-layer KV sharing leads to significant performance degradation. As a result, both methods remain suboptimal for post-training pre-trained LLMs. In this paper, we identify that the \texttt{Softmax} operation is a primary bottleneck for LLM inference and discover that it is actually highly redundant during post-training. We propose Softmax \textbf{Uni}fication in \textbf{Att}e\textbf{n}tion (\textbf{UniAttn}), a novel post-training method that unifies Softmax activations across transformer blocks to reduce LLM inference costs. Additionally, UniAttn adopts a linear projection to compensate for the errors induced by Softmax unification. Experiments show that UniAttn matches the performance of standard post-training while significantly reducing inference costs, outperforming existing efficient architectures during post-training. Our code will be available at \url{https://github.com/Bostoncake/UniAttn}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00415v1">MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 25 pages, 7 figures, Under review at Financial Innovation (FIN)
    </div>
    <details class="paper-abstract">
      MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00406v1">ALU: Agentic LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Information removal or suppression in large language models (LLMs) is a desired functionality, useful in AI regulation, legal compliance, safety, and privacy. LLM unlearning methods aim to remove information on demand from LLMs. Current LLM unlearning methods struggle to balance the unlearning efficacy and utility due to the competing nature of these objectives. Keeping the unlearning process computationally feasible without assuming access to the model weights is an overlooked area. We present the first agentic LLM unlearning (ALU) method, a multi-agent, retrain-free, model-agnostic approach to LLM unlearning that achieves effective unlearning while preserving the utility. Our ALU framework unlearns by involving multiple LLM agents, each designed for a specific step in the unlearning process, without the need to update model weights for any of the agents in the framework. Users can easily request any set of unlearning instances in any sequence, and ALU seamlessly adapts in real time. This is facilitated without requiring any changes in the underlying LLM model. Through extensive experiments on established benchmarks (TOFU, WMDP, WPU) and jailbreaking techniques (many shot, target masking, other languages), we demonstrate that ALU consistently stands out as the most robust LLM unlearning framework among current state-of-the-art methods while incurring a low constant-time cost. We further highlight ALU's superior performance compared to existing methods when evaluated at scale. Specifically, ALU is assessed on up to 1000 unlearning targets, exceeding the evaluation scope of all previously proposed LLM unlearning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00350v1">OrcaLoca: An LLM Agent Framework for Software Issue Localization</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00339v1">Challenges and Innovations in LLM-Powered Fake News Detection: A Synthesis of Approaches and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      The pervasiveness of the dissemination of fake news through social media platforms poses critical risks to the trust of the general public, societal stability, and democratic institutions. This challenge calls for novel methodologies in detection, which can keep pace with the dynamic and multi-modal nature of misinformation. Recent works include powering the detection using large language model advances in multimodal frameworks, methodologies using graphs, and adversarial training in the literature of fake news. Based on the different approaches which can bring success, some key highlights will be underlined: enhanced LLM-improves accuracy through more advanced semantics and cross-modality fusion for robust detections. The review further identifies critical gaps in adaptability to dynamic social media trends, real-time, and cross-platform detection capabilities, as well as the ethical challenges thrown up by the misuse of LLMs. Future directions underline the development of style-agnostic models, cross-lingual detection frameworks, and robust policies with a view to mitigating LLM-driven misinformation. This synthesis thus lays a concrete foundation for those researchers and practitioners committed to reinforcing fake news detection systems with complications that keep on growing in the digital landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v1">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      To reduce memory costs in long-context inference with Large Language Models (LLMs), many recent works focus on compressing the key-value (KV) cache of different tokens. However, we identify that the previous KV cache compression methods measure token importance individually, neglecting the dependency between different tokens in the real-world language characterics. In light of this, we introduce ChunkKV, grouping the tokens in a chunk as a basic compressing unit, and retaining the most informative semantic chunks while discarding the less important ones. Furthermore, observing that ChunkKV exhibits higher similarity in the preserved indices across different layers, we propose layer-wise index reuse to further reduce computational overhead. We evaluated ChunkKV on cutting-edge long-context benchmarks including LongBench and Needle-In-A-HayStack, as well as the GSM8K and JailbreakV in-context learning benchmark. Our experiments with instruction tuning and multi-step reasoning (O1 and R1) LLMs, achieve up to 10\% performance improvement under aggressive compression ratios compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v1">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have seen remarkable advancements and have been extensively integrated across various fields. Despite their progress, LLMs are prone to hallucinations, producing responses that may not be dependable if the models lack sufficient grounding knowledge. To mitigate this issue, methods for estimating uncertainty have been adopted, with a focus on critical tokens as indicators of reliability. Nevertheless, probability-based approaches have shown limitations in assessing token-level reliability due to the erosion of evidence strength information acquired during training. In this paper, we introduce Logits-induced Token Uncertainty (LogU), a novel framework designed to estimate token-specific uncertainty in LLMs in real time, without the need for multiple sampling rounds. By leveraging evidence modeling for the implementation of LogU, we utilize the derived uncertainty measures to steer downstream tasks. Our experimental findings highlight the substantial effectiveness and potential of LogU, marking a significant advancement in addressing the challenge of model hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00258v1">ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance in natural language processing tasks, yet their massive size makes serving them inefficient and costly. Semi-structured pruning has emerged as an effective method for model acceleration, but existing approaches are suboptimal because they focus on local, layer-wise optimizations using heuristic rules, failing to leverage global feedback. We present ProxSparse, a learning-based framework for mask selection enabled by regularized optimization. ProxSparse transforms the rigid, non-differentiable mask selection process into a smoother optimization procedure, allowing gradual mask exploration with flexibility. ProxSparse does not involve additional weight updates once the mask is determined. Our extensive evaluations on 7 widely used models show that ProxSparse consistently outperforms previously proposed semi-structured mask selection methods with significant improvement, demonstrating the effectiveness of our learned approach towards semi-structured pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14654v3">Comparative Analysis of Pooling Mechanisms in LLMs: A Sentiment Analysis Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 Accepted to ISMSI'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP) by delivering state-of-the-art performance across a variety of tasks. Among these, Transformer-based models like BERT and GPT rely on pooling layers to aggregate token-level embeddings into sentence-level representations. Common pooling mechanisms such as Mean, Max, and Weighted Sum play a pivotal role in this aggregation process. Despite their widespread use, the comparative performance of these strategies on different LLM architectures remains underexplored. To address this gap, this paper investigates the effects of these pooling mechanisms on two prominent LLM families -- BERT and GPT, in the context of sentence-level sentiment analysis. Comprehensive experiments reveal that each pooling mechanism exhibits unique strengths and weaknesses depending on the task's specific requirements. Our findings underline the importance of selecting pooling methods tailored to the demands of particular applications, prompting a re-evaluation of common assumptions regarding pooling operations. By offering actionable insights, this study contributes to the optimization of LLM-based models for downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v2">HALO: Hadamard-Assisted Lower-Precision Optimization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which can have large weight and activation outlier values that make lower-precision optimization difficult. To address this, we present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, which mitigate outliers, 2) high-performance kernel support, and 3) FSDP integration for low-precision communication. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.41x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. HALO efficiently supports both standard and parameterefficient fine-tuning (PEFT). Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in 8-bit precision, while delivering performance benefits. Code is available at \url{https://github.com/IST-DASLab/HALO}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v2">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-02-01
      | 💬 16 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
