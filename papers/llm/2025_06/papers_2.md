# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22557v1">MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v1">Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Grokking, i.e., test performance keeps improving long after training loss converged, has been recently witnessed in neural network training, making the mechanism of generalization and other emerging capabilities such as reasoning mysterious. While prior studies usually train small models on a few toy or highly-specific tasks for thousands of epochs, we conduct the first study of grokking on checkpoints during one-pass pretraining of a 7B large language model (LLM), i.e., OLMoE. We compute the training loss and evaluate generalization on diverse benchmark tasks, including math reasoning, code generation, and commonsense/domain-specific knowledge retrieval tasks. Our study, for the first time, verifies that grokking still happens in the pretraining of large-scale foundation models, though different data may enter grokking stages asynchronously. We further demystify grokking's "emergence of generalization" by investigating LLM internal dynamics. Specifically, we find that training samples' pathways (i.e., expert choices across layers) evolve from random, instance-specific to more structured and shareable between samples during grokking. Also, the complexity of a sample's pathway reduces despite the converged loss. These indicate a memorization-to-generalization conversion, providing a mechanistic explanation of delayed generalization. In the study, we develop two novel metrics to quantify pathway distance and the complexity of a single pathway. We show their ability to predict the generalization improvement on diverse downstream tasks. They are efficient, simple to compute and solely dependent on training data. Hence, they have practical value for pretraining, enabling us to monitor the generalization performance without finetuning and test. Theoretically, we show that more structured pathways reduce model complexity and improve the generalization bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21497v1">Enhancing User Engagement in Socially-Driven Dialogue through Interactive LLM Alignments</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Enhancing user engagement through interactions plays an essential role in socially-driven dialogues. While prior works have optimized models to reason over relevant knowledge or plan a dialogue act flow, the relationship between user engagement and knowledge or dialogue acts is subtle and does not guarantee user engagement in socially-driven dialogues. To this end, we enable interactive LLMs to learn user engagement by leveraging signals from the future development of conversations. Specifically, we adopt a more direct and relevant indicator of user engagement, i.e., the user's reaction related to dialogue intention after the interaction, as a reward to align interactive LLMs. To achieve this, we develop a user simulator to interact with target interactive LLMs and explore interactions between the user and the interactive LLM system via \textit{i$\times$MCTS} (\textit{M}onte \textit{C}arlo \textit{T}ree \textit{S}earch for \textit{i}nteraction). In this way, we collect a dataset containing pairs of higher and lower-quality experiences using \textit{i$\times$MCTS}, and align interactive LLMs for high-level user engagement by direct preference optimization (DPO) accordingly. Experiments conducted on two socially-driven dialogue scenarios (emotional support conversations and persuasion for good) demonstrate that our method effectively enhances user engagement in interactive LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02398v3">Prompting with Phonemes: Enhancing LLMs' Multilinguality for Non-Latin Script Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted to NAACL 2025 (Main Conference). This version contains minor improvements to the camera-ready
    </div>
    <details class="paper-abstract">
      Although multilingual LLMs have achieved remarkable performance across benchmarks, we find they continue to underperform on non-Latin script languages across contemporary LLM families. This discrepancy arises from the fact that LLMs are pretrained with orthographic scripts, which are dominated by Latin characters that obscure their shared phonology with non-Latin scripts. We propose leveraging phonemic transcriptions as complementary signals to induce script-invariant representations. Our study demonstrates that integrating phonemic signals improves performance across both non-Latin and Latin script languages, with a particularly significant impact on closing the performance gap between the two. Through detailed experiments, we show that phonemic and orthographic scripts retrieve distinct examples for in-context learning (ICL). This motivates our proposed Mixed-ICL retrieval strategy, where further aggregation from both leads to our significant performance improvements for both Latin script languages (up to 12.6%) and non-Latin script languages (up to 15.1%) compared to randomized ICL retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21443v1">Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15830v2">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v3">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21384v1">Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted at SIGIR 2025 LiveRAG Workshop (Oral Presentation)
    </div>
    <details class="paper-abstract">
      Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21319v1">Multimodal LLMs for Visualization Reconstruction and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Visualizations are crucial for data communication, yet understanding them requires comprehension of both visual elements and their underlying data relationships. Current multimodal large models, while effective in natural image understanding, struggle with visualization due to their inability to decode the data-to-visual mapping rules and extract structured information. To address these challenges, we present a novel dataset and train multimodal visualization LLMs specifically designed for understanding. Our approach combines chart images with their corresponding vectorized representations, encoding schemes, and data features. The proposed vector format enables compact and accurate reconstruction of visualization content. Experimental results demonstrate significant improvements in both data extraction accuracy and chart reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13379v2">Thinkless: LLM Learns When to Think</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21285v1">Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21199v1">MedPrompt: LLM-CNN Fusion with Weight Routing for Medical Image Segmentation and Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 40 pages, 8 Tables, 9 Figures
    </div>
    <details class="paper-abstract">
      Current medical image analysis systems are typically task-specific, requiring separate models for classification and segmentation, and lack the flexibility to support user-defined workflows. To address these challenges, we introduce MedPrompt, a unified framework that combines a few-shot prompted Large Language Model (Llama-4-17B) for high-level task planning with a modular Convolutional Neural Network (DeepFusionLab) for low-level image processing. The LLM interprets user instructions and generates structured output to dynamically route task-specific pretrained weights. This weight routing approach avoids retraining the entire framework when adding new tasks-only task-specific weights are required, enhancing scalability and deployment. We evaluated MedPrompt across 19 public datasets, covering 12 tasks spanning 5 imaging modalities. The system achieves a 97% end-to-end correctness in interpreting and executing prompt-driven instructions, with an average inference latency of 2.5 seconds, making it suitable for near real-time applications. DeepFusionLab achieves competitive segmentation accuracy (e.g., Dice 0.9856 on lungs) and strong classification performance (F1 0.9744 on tuberculosis). Overall, MedPrompt enables scalable, prompt-driven medical imaging by combining the interpretability of LLMs with the efficiency of modular CNNs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00753v4">LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21071v1">Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 20 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Teaching large language models (LLMs) to use tools is crucial for improving their problem-solving abilities and expanding their applications. However, effectively using tools is challenging because it requires a deep understanding of tool functionalities and user intentions. Previous methods relied mainly on LLMs to generate instruction data, but the quality of these data was often insufficient. In this paper, we propose a new method that uses knowledge graphs to generate high-quality instruction data for LLMs. Knowledge graphs are manually curated datasets rich in semantic information. We begin by extracting various query pathways from a given knowledge graph, which are transformed into a broad spectrum of user queries. We then translate the relationships between entities into actionable tools and parse the pathways of each query into detailed solution steps, thereby creating high-quality instruction data. Our experiments show that fine-tuning on just a small sample of this synthetic data can significantly improve the tool utilization and overall capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v3">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21033v1">BLOCKS: Blockchain-supported Cross-Silo Knowledge Sharing for Efficient LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      The hallucination problem of Large Language Models (LLMs) has increasingly drawn attention. Augmenting LLMs with external knowledge is a promising solution to address this issue. However, due to privacy and security concerns, a vast amount of downstream task-related knowledge remains dispersed and isolated across various "silos," making it difficult to access. To bridge this knowledge gap, we propose a blockchain-based external knowledge framework that coordinates multiple knowledge silos to provide reliable foundational knowledge for large model retrieval while ensuring data security. Technically, we distill knowledge from local data into prompts and execute transactions and records on the blockchain. Additionally, we introduce a reputation mechanism and cross-validation to ensure knowledge quality and provide incentives for participation. Furthermore, we design a query generation framework that provides a direct API interface for large model retrieval. To evaluate the performance of our proposed framework, we conducted extensive experiments on various knowledge sources. The results demonstrate that the proposed framework achieves efficient LLM service knowledge sharing in blockchain environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18389v3">ALLSTaR: Automated LLM-Driven Scheduler Generation and Testing for Intent-Based RAN</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Under submission to an IEEE journal, copyright may change without notice
    </div>
    <details class="paper-abstract">
      The evolution toward open, programmable O-RAN and AI-RAN 6G networks creates unprecedented opportunities for Intent-Based Networking (IBN) to dynamically optimize RAN[...]. However, applying IBN effectively to the RAN scheduler [...] remains a significant challenge. Current approaches predominantly rely on coarse-grained network slicing, lacking the granularity for dynamic adaptation to individual user conditions and traffic patterns. Despite the existence of a vast body of scheduling algorithms [...], their practical utilization is hindered by implementation heterogeneity, insufficient systematic evaluation in production environments, and the complexity of developing high-performance scheduler implementations.[...] To address these limitations, we propose ALLSTaR (Automated LLm-driven Scheduler generation and Testing for intent-based RAN), a novel framework leveraging LLMs for automated, intent-driven scheduler design, implementation, and evaluation. ALLSTaR interprets NL intents, automatically generates functional scheduler code from the research literature using OCR and LLMs, and intelligently matches operator intents to the most suitable scheduler(s). Our implementation deploys these schedulers as O-RAN dApps, enabling on-the-fly deployment and testing on a production-grade, 5G-compliant testbed. This approach has enabled the largest-scale OTA experimental comparison of 18 scheduling algorithms automatically synthesized from the academic literature. The resulting performance profiles serve as the input for our Intent-Based Scheduling (IBS) framework, which dynamically selects and deploys appropriate schedulers that optimally satisfy operator intents. We validate our approach through multiple use cases unattainable with current slicing-based optimization techniques, demonstrating fine-grained control based on buffer status, physical layer conditions, and heterogeneous traffic types
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20993v1">SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18313v2">Will LLMs be Professional at Fund Investment? DeepFund: A Live Arena Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 6 pages, 3 figures, perspective paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, but their effectiveness in financial decision-making remains inadequately evaluated. Current benchmarks primarily assess LLMs' understanding on financial documents rather than the ability to manage assets or dig out trading opportunities in dynamic market conditions. Despite the release of new benchmarks for evaluating diversified tasks on the financial domain, we identified four major problems in these benchmarks, which are data leakage, navel-gazing, over-intervention, and maintenance-hard. To pave the research gap, we introduce DeepFund, a comprehensive arena platform for evaluating LLM-based trading strategies in a live environment. Our approach implements a multi-agent framework where they serve as multiple key roles that realize the real-world investment decision processes. Moreover, we provide a web interface that visualizes LLMs' performance with fund investment metrics across different market conditions, enabling detailed comparative analysis. Through DeepFund, we aim to provide a more realistic and fair assessment on LLM's capabilities in fund investment, offering diversified insights and revealing their potential applications in real-world financial markets. Our code is publicly available at https://github.com/HKUSTDial/DeepFund.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03359v2">WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at https://whoisspy.ai/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20982v1">Our Coding Adventure: Using LLMs to Personalise the Narrative of a Tangible Programming Robot for Preschoolers</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 accepted at D-SAIL Workshop - Transformative Curriculum Design: Digitalization, Sustainability, and AI Literacy for 21st Century Learning
    </div>
    <details class="paper-abstract">
      Finding balanced ways to employ Large Language Models (LLMs) in education is a challenge due to inherent risks of poor understanding of the technology and of a susceptible audience. This is particularly so with younger children, who are known to have difficulties with pervasive screen time. Working with a tangible programming robot called Cubetto, we propose an approach to benefit from the capabilities of LLMs by employing such models in the preparation of personalised storytelling, necessary for preschool children to get accustomed to the practice of commanding the robot. We engage in action research to develop an early version of a formalised process to rapidly prototype game stories for Cubetto. Our approach has both reproducible results, because it employs open weight models, and is model-agnostic, because we test it with 5 different LLMs. We document on one hand the process, the used materials and prompts, and on the other the learning experience and outcomes. We deem the generation successful for the intended purposes of using the results as a teacher aid. Testing the models on 4 different task scenarios, we encounter issues of consistency and hallucinations and document the corresponding evaluation process and attempts (some successful and some not) to overcome these issues. Importantly, the process does not expose children to LLMs directly. Rather, the technology is used to help teachers easily develop personalised narratives on children's preferred topics. We believe our method is adequate for preschool classes and we are planning to further experiment in real-world educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19324v3">Reward-Guided Speculative Decoding for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. The code is available at https://github.com/BaohaoLiao/RSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20938v1">ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      GPGPU architectures have become significantly diverse in recent years, which has led to an emergence of a variety of specialized programming models and software stacks to support them. While portable execution models exist, they still require significant developer effort to port to and optimize for different hardware architectures. Recent advances in large language models (LLMs) can help us reduce some of this programmer burden. In this paper, we present a novel benchmark and testing framework, ParEval-Repo, which can be used to evaluate the efficacy of LLM-based approaches in automatically translating entire codebases across GPGPU execution models. ParEval-Repo includes several scientific computing and AI mini-applications in a range of programming models, and levels of repository complexity. We use ParEval-Repo to evaluate a range of state-of-the-art open-source and commercial LLMs, with both a non-agentic and a top-down agentic approach. We assess code generated by the LLMs and approaches in terms of compilability, functional correctness, categories of build errors, and the cost of translation in terms of the number of inference tokens. Our results demonstrate that LLM translation of scientific applications is feasible for small programs but difficulty with generating functional build systems and cross-file dependencies pose challenges in scaling to larger codebases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13547v2">ToolScan: A Benchmark for Characterizing Errors in Tool-Use LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) is one of the most critical aspects of building a performant compound AI system. Since the output from LLMs propagate to downstream steps, identifying LLM errors is crucial to system performance. A common task for LLMs in AI systems is tool use. While there are several benchmark environments for evaluating LLMs on this task, they typically only give a success rate without any explanation of the failure cases. To solve this problem, we introduce TOOLSCAN, a new benchmark to identify error patterns in LLM output on tool-use tasks. Our benchmark data set comprises of queries from diverse environments that can be used to test for the presence of seven newly characterized error patterns. Using TOOLSCAN, we show that even the most prominent LLMs exhibit these error patterns in their outputs. Researchers can use these insights from TOOLSCAN to guide their error mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20921v1">LLM-guided Chemical Process Optimization with a Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 16 pages (main manuscript without references), 2 figures
    </div>
    <details class="paper-abstract">
      Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v2">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22516v1">Can "consciousness" be observed from large language model (LLM) internal states? Dissecting LLM representations obtained from Theory of Mind test with Integrated Information Theory and Span Representation analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Published as a journal paper at: https://doi.org/10.1016/j.nlp.2025.100163
    </div>
    <details class="paper-abstract">
      Integrated Information Theory (IIT) provides a quantitative framework for explaining consciousness phenomenon, positing that conscious systems comprise elements integrated through causal properties. We apply IIT 3.0 and 4.0 -- the latest iterations of this framework -- to sequences of Large Language Model (LLM) representations, analyzing data derived from existing Theory of Mind (ToM) test results. Our study systematically investigates whether the differences of ToM test performances, when presented in the LLM representations, can be revealed by IIT estimates, i.e., $\Phi^{\max}$ (IIT 3.0), $\Phi$ (IIT 4.0), Conceptual Information (IIT 3.0), and $\Phi$-structure (IIT 4.0). Furthermore, we compare these metrics with the Span Representations independent of any estimate for consciousness. This additional effort aims to differentiate between potential "consciousness" phenomena and inherent separations within LLM representational space. We conduct comprehensive experiments examining variations across LLM transformer layers and linguistic spans from stimuli. Our results suggest that sequences of contemporary Transformer-based LLM representations lack statistically significant indicators of observed "consciousness" phenomena but exhibit intriguing patterns under $\textit{spatio}$-permutational analyses. The Appendix and code are available as Supplementary Materials at: https://doi.org/10.1016/j.nlp.2025.100163.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20666v1">Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20598v1">Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03905v3">Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 25 pages, 12 images, 10 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      LLMs have garnered considerable attention for their potential to streamline Automated Program Repair (APR). LLM-based approaches can either insert the correct code or directly generate patches when provided with buggy methods. However, most of LLM-based APR methods rely on a single type of software information, without fully leveraging different software artifacts. Despite this, many LLM-based approaches do not explore which specific types of information best assist in APR. Addressing this gap is crucial for advancing LLM-based APR techniques. We propose DEVLoRe to use issue content (description and message) and stack error traces to localize buggy methods, then rely on debug information in buggy methods and issue content and stack error to localize buggy lines and generate plausible patches which can pass all unit tests. The results show that while issue content is particularly effective in assisting LLMs with fault localization and program repair, different types of software artifacts complement each other. By incorporating different artifacts, DEVLoRe successfully locates 49.3% and 47.6% of single and non-single buggy methods and generates 56.0% and 14.5% plausible patches for the Defects4J v2.0 dataset, respectively. This outperforms current state-of-the-art APR methods. Furthermore, we re-implemented and evaluated our framework, demonstrating its effectiveness in its effectiveness in resolving 9 unique issues compared to other state-of-the-art frameworks using the same or more advanced models on SWE-bench Lite.We also discussed whether a leading framework for Python code can be directly applied to Java code, or vice versa. The source code and experimental results of this work for replication are available at https://github.com/XYZboom/DEVLoRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20544v1">When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting. Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20451v1">Automatic Demonstration Selection for LLM-based Tabular Data Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      A fundamental question in applying In-Context Learning (ICL) for tabular data classification is how to determine the ideal number of demonstrations in the prompt. This work addresses this challenge by presenting an algorithm to automatically select a reasonable number of required demonstrations. Our method distinguishes itself by integrating not only the tabular data's distribution but also the user's selected prompt template and the specific Large Language Model (LLM) into its estimation. Rooted in Spectral Graph Theory, our proposed algorithm defines a novel metric to quantify the similarities between different demonstrations. We then construct a similarity graph and analyze the eigenvalues of its Laplacian to derive the minimum number of demonstrations capable of representing the data within the LLM's intrinsic representation space. We validate the efficacy of our approach through experiments comparing its performance against conventional random selection algorithms on diverse datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11677v2">Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive performance across diverse tasks but often struggle to accurately gauge their knowledge boundaries, leading to confident yet incorrect responses. This paper explores leveraging LLMs' internal states to enhance their perception of knowledge boundaries from efficiency and risk perspectives. We investigate whether LLMs can estimate their confidence using internal states before response generation, potentially saving computational resources. Our experiments on datasets like Natural Questions, HotpotQA, and MMLU reveal that LLMs demonstrate significant pre-generation perception, which is further refined post-generation, with perception gaps remaining stable across varying conditions. To mitigate risks in critical domains, we introduce Confidence Consistency-based Calibration ($C^3$), which assesses confidence consistency through question reformulation. $C^3$ significantly improves LLMs' ability to recognize their knowledge gaps, enhancing the unknown perception rate by 5.6% on NQ and 4.9% on HotpotQA. Our findings suggest that pre-generation confidence estimation can optimize efficiency, while $C^3$ effectively controls output risks, advancing the reliability of LLMs in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17219v2">No Free Lunch: Rethinking Internal Feedback for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning has emerged as a powerful paradigm for post-training large language models (LLMs) to improve reasoning. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) have shown strong results, but they require extensive external supervision. We investigate an alternative class of methods, Reinforcement Learning from Internal Feedback (RLIF), which relies solely on intrinsic model-derived signals instead of external rewards. In particular, we leverage unsupervised reward proxies such as token-level entropy, trajectory-level entropy, and self-certainty. Our theoretical analysis shows these internal objectives are partially equivalent, and we empirically evaluate various RLIF strategies on challenging math reasoning benchmarks. Experimental results demonstrate that RLIF can boost the reasoning performance of base LLMs at the beginning phase of the training, matching or surpassing RLVR techniques on these tasks. However, when training progresses, performance degrades even below the model before training. Moreover, we find that RLIF yields little improvement for instruction-tuned models, indicating diminishing returns of intrinsic feedback once an LLM is already instruction-tuned. We further analyze this limitation by mixing model weights and explain the reason of RLIF's training behaviors, providing practical guidelines for integrating internal feedback signals into LLM training. We hope our analysis of internal feedback will inform more principled and effective strategies for LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20353v1">DipSVD: Dual-importance Protected SVD for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      The ever-increasing computational demands and deployment costs of large language models (LLMs) have spurred numerous compressing methods. Compared to quantization and unstructured pruning, SVD compression offers superior hardware compatibility and theoretical guarantees. However, existing SVD-based methods focus on the overall discrepancy between the original and compressed matrices while overlooking the protection of critical components within the matrix, which leads to inferior performance in the compressed models. This paper proposes a dual-level importance protection mechanism to enhance SVD-based compression methods: (1) local importance protection: preserving the most critical singular vectors within each weight matrix through channel-weighted data whitening; and (2) global importance protection: enabling less important layers to bear a greater portion of the compression burden through either a heuristic or optimization-based approach, thereby minimizing the impact of compression on critical layers. Extensive experiments demonstrate that DipSVD outperforms existing SVD-based compression approaches across multiple benchmarks, achieving superior model performance especially at high model compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02502v2">LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL 2025, our code is available at https://github.com/ZNLP/LADM
    </div>
    <details class="paper-abstract">
      Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20194v1">DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver strong performance but are difficult to deploy due to high memory and compute costs. While pruning reduces these demands, most methods ignore activation sparsity observed at runtime. We reinterpret activation sparsity as dynamic structured weight sparsity and propose DuoGPT, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity. To preserve accuracy, we extend the Optimal Brain Compression (OBC) framework with activation-aware calibration and introduce output residuals from the dense model as correction terms. We further optimize the solution for efficient GPU execution, enabling scalability to billion-parameter LLMs. Evaluations on LLaMA-2 and LLaMA-3 show that DuoGPT outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39$\times$ compared to the baseline dense model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20187v1">Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 15 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Advanced Large Language Models (LLMs) have achieved impressive performance across a wide range of complex and long-context natural language tasks. However, performing long-context LLM inference locally on a commodity GPU (a PC) with privacy concerns remains challenging due to the increasing memory demands of the key-value (KV) cache. Existing systems typically identify important tokens and selectively offload their KV data to GPU and CPU memory. The KV data needs to be offloaded to disk due to the limited memory on a commodity GPU, but the process is bottlenecked by token importance evaluation overhead and the disk's low bandwidth. In this paper, we present LeoAM, the first efficient importance-aware long-context LLM inference system for a single commodity GPU with adaptive hierarchical GPU-CPU-Disk KV management. Our system employs an adaptive KV management strategy that partitions KV data into variable-sized chunks based on the skewed distribution of attention weights across different layers to reduce computational and additional transmission overheads. Moreover, we propose a lightweight KV abstract method, which minimizes transmission latency by storing and extracting the KV abstract of each chunk on disk instead of the full KV data. LeoAM also leverages the dynamic compression and pipeline techniques to further accelerate inference. Experimental results demonstrate that LongInfer achieves an average inference latency speedup of 3.46x, while maintaining comparable LLM response quality. In scenarios with larger batch sizes, it achieves up to a 5.47x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20170v1">JsDeObsBench: Measuring and Benchmarking LLMs for JavaScript Deobfuscation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted by ACM CCS 2025
    </div>
    <details class="paper-abstract">
      Deobfuscating JavaScript (JS) code poses a significant challenge in web security, particularly as obfuscation techniques are frequently used to conceal malicious activities within scripts. While Large Language Models (LLMs) have recently shown promise in automating the deobfuscation process, transforming detection and mitigation strategies against these obfuscated threats, a systematic benchmark to quantify their effectiveness and limitations has been notably absent. To address this gap, we present JsDeObsBench, a dedicated benchmark designed to rigorously evaluate the effectiveness of LLMs in the context of JS deobfuscation. We detail our benchmarking methodology, which includes a wide range of obfuscation techniques ranging from basic variable renaming to sophisticated structure transformations, providing a robust framework for assessing LLM performance in real-world scenarios. Our extensive experimental analysis investigates the proficiency of cutting-edge LLMs, e.g., GPT-4o, Mixtral, Llama, and DeepSeek-Coder, revealing superior performance in code simplification despite challenges in maintaining syntax accuracy and execution reliability compared to baseline methods. We further evaluate the deobfuscation of JS malware to exhibit the potential of LLMs in security scenarios. The findings highlight the utility of LLMs in deobfuscation applications and pinpoint crucial areas for further improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16789v2">Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 8 pages, ACL style
    </div>
    <details class="paper-abstract">
      Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20167v1">SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00845v2">Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted to KDD 2025 Research Track
    </div>
    <details class="paper-abstract">
      Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20128v1">CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted at LLM4Eval @ SIGIR 2025
    </div>
    <details class="paper-abstract">
      RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19262v2">What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13819v2">LLM-Empowered IoT for 6G Networks: Architecture, Challenges, and Solutions</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted for publication in IEEE Internet of Things Magazine. \c{opyright} 2025 IEEE. The final version will be published in IEEE Xplore
    </div>
    <details class="paper-abstract">
      The Internet of Things (IoT) in the sixth generation (6G) era is envisioned to evolve towards intelligence, ubiquity, and self-optimization. Large language models (LLMs) have demonstrated remarkable generalization capabilities across diverse domains, including natural language processing (NLP), computer vision (CV), and beyond. In this article, we propose an LLM-empowered IoT architecture for 6G networks to achieve intelligent autonomy while supporting advanced IoT applications. LLMs are pushed to the edge of the 6G network to support the synergy of LLMs and IoT. LLM solutions are tailored to both IoT application requirements and IoT management needs, i.e., LLM for IoT. On the other hand, edge inference and edge fine-tuning are discussed to support the deployment of LLMs, i.e., LLM on IoT. Furthermore, we propose a memory-efficient split federated learning (SFL) framework for LLM fine-tuning on heterogeneous IoT devices that alleviates memory pressures on both IoT devices and the edge server while achieving comparable performance and convergence time. Finally, a case study is presented, followed by a discussion about open issues of LLM-empowered IoT for 6G networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02244v2">Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CBT</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being used as ad-hoc therapists. Research suggests that LLMs outperform human counselors when generating a single, isolated empathetic response; however, their session-level behavior remains understudied. In this study, we compare the session-level behaviors of human counselors with those of an LLM prompted by a team of peer counselors to deliver single-session Cognitive Behavioral Therapy (CBT). Our three-stage, mixed-methods study involved: a) a year-long ethnography of a text-based support platform where seven counselors iteratively refined CBT prompts through self-counseling and weekly focus groups; b) the manual simulation of human counselor sessions with a CBT-prompted LLM, given the full patient dialogue and contextual notes; and c) session evaluations of both human and LLM sessions by three licensed clinical psychologists using CBT competence measures. Our results show a clear trade-off. Human counselors excel at relational strategies -- small talk, self-disclosure, and culturally situated language -- that lead to higher empathy, collaboration, and deeper user reflection. LLM counselors demonstrate higher procedural adherence to CBT techniques but struggle to sustain collaboration, misread cultural cues, and sometimes produce "deceptive empathy," i.e., formulaic warmth that can inflate users' expectations of genuine human care. Taken together, our findings imply that while LLMs might outperform counselors in generating single empathetic responses, their ability to lead sessions is more limited, highlighting that therapy cannot be reduced to a standalone natural language processing (NLP) task. We call for carefully designed human-AI workflows in scalable support: LLMs can scaffold evidence-based techniques, while peers provide relational support. We conclude by mapping concrete design opportunities and ethical guardrails for such hybrid systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v2">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08400v2">mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 working paper
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20073v1">A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13305v3">Computation Mechanism Behind LLM Position Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL 2025 Main Long Paper
    </div>
    <details class="paper-abstract">
      Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15928v2">Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Under review for KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20886v1">Omniwise: Predicting GPU Kernels Performance with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20856v1">Leaner Training, Lower Leakage: Revisiting Memorization in LLM Fine-Tuning with LoRA</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method. In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20822v1">Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20807v1">GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 4 page paper plus Appendices. Accepted to the ES-FoMo "Efficient Systems for Foundation Models" workshop at ICML 2025
    </div>
    <details class="paper-abstract">
      Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels. Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise. Since quantitative results from an ongoing performance competition were embargoed on paper submission date, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly evolving hardware environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20803v1">The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 main paper is 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14133v2">GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 38 pages, 8 tables, 18 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown impressive capabilities across various natural language processing tasks, yet remain vulnerable to input prompts, known as jailbreak attacks, carefully designed to bypass safety guardrails and elicit harmful responses. Traditional methods rely on manual heuristics but suffer from limited generalizability. Despite being automatic, optimization-based attacks often produce unnatural prompts that can be easily detected by safety filters or require high computational costs due to discrete token optimization. In this paper, we introduce Generative Adversarial Suffix Prompter (GASP), a novel automated framework that can efficiently generate human-readable jailbreak prompts in a fully black-box setting. In particular, GASP leverages latent Bayesian optimization to craft adversarial suffixes by efficiently exploring continuous latent embedding spaces, gradually optimizing the suffix prompter to improve attack efficacy while balancing prompt coherence via a targeted iterative refinement procedure. Through comprehensive experiments, we show that GASP can produce natural adversarial prompts, significantly improving jailbreak success over baselines, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05199v3">CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinement</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation but require significant resources and often over-generalize, limiting their task-specific efficiency. Fine-tuning smaller, open-source LLMs provides a cost-effective alternative. However, standard supervised approaches rely only on correct examples, missing valuable insights from failures. We introduce CodeLutra, a framework that leverages both correct and incorrect code attempts. Instead of using only correct solutions, CodeLutra applies iterative preference-based refinement, comparing successful and failed outputs to better approximate desired results. This approach narrows the performance gap with state-of-the-art larger models without requiring massive datasets or auxiliary models. For instance, on a challenging data science coding task, using only 500 samples improved Llama-3-8B's accuracy from 28.2% to 48.6%, approaching GPT-4's level. By learning from both successes and mistakes, CodeLutra provides a scalable and efficient path to high-quality code generation, making smaller open-source models more competitive with leading closed-source alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20743v1">A Survey of AI for Materials Science: Foundation Models, LLM Agents, Datasets, and Tools</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Foundation models (FMs) are catalyzing a transformative shift in materials science (MatSci) by enabling scalable, general-purpose, and multimodal AI systems for scientific discovery. Unlike traditional machine learning models, which are typically narrow in scope and require task-specific engineering, FMs offer cross-domain generalization and exhibit emergent capabilities. Their versatility is especially well-suited to materials science, where research challenges span diverse data types and scales. This survey provides a comprehensive overview of foundation models, agentic systems, datasets, and computational tools supporting this growing field. We introduce a task-driven taxonomy encompassing six broad application areas: data extraction, interpretation and Q\&A; atomistic simulation; property prediction; materials structure, design and discovery; process planning, discovery, and optimization; and multiscale modeling. We discuss recent advances in both unimodal and multimodal FMs, as well as emerging large language model (LLM) agents. Furthermore, we review standardized datasets, open-source tools, and autonomous experimental platforms that collectively fuel the development and integration of FMs into research workflows. We assess the early successes of foundation models and identify persistent limitations, including challenges in generalizability, interpretability, data imbalance, safety concerns, and limited multimodal fusion. Finally, we articulate future research directions centered on scalable pretraining, continual learning, data governance, and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06877v2">Right Is Not Enough: The Pitfalls of Outcome Supervision in Training LLMs for Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Outcome-rewarded Large Language Models (LLMs) have demonstrated remarkable success in mathematical problem-solving. However, this success often masks a critical issue: models frequently achieve correct answers through fundamentally unsound reasoning processes, a phenomenon indicative of reward hacking. We introduce MathOlympiadEval, a new dataset with fine-grained annotations, which reveals a significant gap between LLMs' answer correctness and their low process correctness. Existing automated methods like LLM-as-a-judge struggle to reliably detect these reasoning flaws. To address this, we propose ParaStepVerifier, a novel methodology for meticulous, step-by-step verification of mathematical solutions. ParaStepVerifier identifies incorrect reasoning steps. Empirical results demonstrate that ParaStepVerifier substantially improves the accuracy of identifying flawed solutions compared to baselines, especially for complex, multi-step problems. This offers a more robust path towards evaluating and training LLMs with genuine mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05153v2">A text-to-tabular approach to generate synthetic patient data using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 12 pages, 3 figures. Accepted to the 2025 IEEE International Conference on Healthcare Informatics (IEEE ICHI 2025), 2025, Rende (CS), Calabria, Italy
    </div>
    <details class="paper-abstract">
      Access to large-scale high-quality healthcare databases is key to accelerate medical research and make insightful discoveries about diseases. However, access to such data is often limited by patient privacy concerns, data sharing restrictions and high costs. To overcome these limitations, synthetic patient data has emerged as an alternative. However, synthetic data generation (SDG) methods typically rely on machine learning (ML) models trained on original data, leading back to the data scarcity problem. We propose an approach to generate synthetic tabular patient data that does not require access to the original data, but only a description of the desired database. We leverage prior medical knowledge and in-context learning capabilities of large language models (LLMs) to generate realistic patient data, even in a low-resource setting. We quantitatively evaluate our approach against state-of-the-art SDG models, using fidelity, privacy, and utility metrics. Our results show that while LLMs may not match the performance of state-of-the-art models trained on the original data, they effectively generate realistic patient data with well-preserved clinical correlations. An ablation study highlights key elements of our prompt contributing to high-quality synthetic patient data generation. This approach, which is easy to use and does not require original data or advanced ML skills, is particularly valuable for quickly generating custom-designed patient data, supporting project implementation and providing educational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19607v1">Correcting Hallucinations in News Summaries: Exploration of Self-Correcting LLM Methods with External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted to FEVER @ ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown remarkable capabilities to generate coherent text, they suffer from the issue of hallucinations -- factually inaccurate statements. Among numerous approaches to tackle hallucinations, especially promising are the self-correcting methods. They leverage the multi-turn nature of LLMs to iteratively generate verification questions inquiring additional evidence, answer them with internal or external knowledge, and use that to refine the original response with the new corrections. These methods have been explored for encyclopedic generation, but less so for domains like news summarization. In this work, we investigate two state-of-the-art self-correcting systems by applying them to correct hallucinated summaries using evidence from three search engines. We analyze the results and provide insights into systems' performance, revealing interesting practical findings on the benefits of search engine snippets and few-shot prompts, as well as high alignment of G-Eval and human evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17728v2">KAG-Thinker: Interactive Thinking and Deep Reasoning in LLMs via Knowledge-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In this paper, we introduce KAG-Thinker, which upgrade KAG to a multi-turn interactive thinking and deep reasoning framework powered by a dedicated parameter-light large language model (LLM). Our approach constructs a structured thinking process for solving complex problems, enhancing the the logical coherence and contextual consistency of the reasoning process in question-answering (Q&A) tasks on domain-specific knowledge bases (KBs) within LLMs. Following the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG, this framework first decomposes complex questions into independently solvable sub-problems (which are also referred to as logical forms) through \textbf{breadth decomposition}. Each such logical form is represented in two equivalent forms-natural language and logical function-and subsequently classified as either a Knowledge Retrieval or Reasoning Analysis task. Dependencies and parameter passing between these tasks are explicitly modeled via logical function interfaces. In the solving process, the Retrieval function performs retrieval tasks. It retrieves one-hop structured and unstructured information of specified knowledge unit. While the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} module to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} module to enhance the comprehensiveness of knowledge acquisition...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19563v1">PrivacyXray: Detecting Privacy Breaches in LLMs through Semantic Consistency and Probability Certainty</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in sensitive domains, including healthcare, finance, and legal services, raising concerns about potential private information leaks during inference. Privacy extraction attacks, such as jailbreaking, expose vulnerabilities in LLMs by crafting inputs that force the models to output sensitive information. However, these attacks cannot verify whether the extracted private information is accurate, as no public datasets exist for cross-validation, leaving a critical gap in private information detection during inference. To address this, we propose PrivacyXray, a novel framework detecting privacy breaches by analyzing LLM inner states. Our analysis reveals that LLMs exhibit higher semantic coherence and probabilistic certainty when generating correct private outputs. Based on this, PrivacyXray detects privacy breaches using four metrics: intra-layer and inter-layer semantic similarity, token-level and sentence-level probability distributions. PrivacyXray addresses critical challenges in private information detection by overcoming the lack of open-source private datasets and eliminating reliance on external data for validation. It achieves this through the synthesis of realistic private data and a detection mechanism based on the inner states of LLMs. Experiments show that PrivacyXray achieves consistent performance, with an average accuracy of 92.69% across five LLMs. Compared to state-of-the-art methods, PrivacyXray achieves significant improvements, with an average accuracy increase of 20.06%, highlighting its stability and practical utility in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v2">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 15 pages, 19 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11558v2">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 I would like to request the withdrawal of this submission because the current version contains significant errors and incomplete results. I intend to revise the manuscript thoroughly before resubmitting. I apologize for the oversight and appreciate your understanding
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with GPT-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19527v1">KnowMap: Efficient Knowledge-Driven Task Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) possess significant capabilities in open-world agent tasks, they also face challenges in rapidly adapting to new, specialized tasks due to their reliance on static pre-trained knowledge. Traditional methods such as fine-tuning are often costly, data-intensive, and may lead to "catastrophic forgetting." Therefore, we present KnowMap, a novel approach that dynamically constructs a knowledge base from environmental and experiential data. KnowMap fine-tunes a small knowledge-embedding model to equip a larger LLM with valuable task-specific knowledge. Our experiments on the ScienceWorld benchmark demonstrate 17.71% improvement for the performance of gpt-4-turbo model. KnowMap not only provides an efficient and effective means for LLM task-adapting, but also highlights how integrating environmental and experiential knowledge can enhance LLMs' reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19525v1">Automatic Posology Structuration : What role for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Automatically structuring posology instructions is essential for improving medication safety and enabling clinical decision support. In French prescriptions, these instructions are often ambiguous, irregular, or colloquial, limiting the effectiveness of classic ML pipelines. We explore the use of Large Language Models (LLMs) to convert free-text posologies into structured formats, comparing prompt-based methods and fine-tuning against a "pre-LLM" system based on Named Entity Recognition and Linking (NERL). Our results show that while prompting improves performance, only fine-tuned LLMs match the accuracy of the baseline. Through error analysis, we observe complementary strengths: NERL offers structural precision, while LLMs better handle semantic nuances. Based on this, we propose a hybrid pipeline that routes low-confidence cases from NERL (<0.8) to the LLM, selecting outputs based on confidence scores. This strategy achieves 91% structuration accuracy while minimizing latency and compute. Our results show that this hybrid approach improves structuration accuracy while limiting computational cost, offering a scalable solution for real-world clinical use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06102v2">SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs) in Business Intelligence (BI), existing solutions face critical challenges in industrial deployments: functionality deficiencies from legacy systems failing to meet evolving LLM-era user demands, interaction limitations from single-round SQL generation paradigms inadequate for multi-round clarification, and cost for domain adaptation arising from cross-domain methods migration. We present SiriusBI, a practical LLM-powered BI system addressing the challenges of industrial deployments through three key innovations: (a) An end-to-end architecture integrating multi-module coordination to overcome functionality gaps in legacy systems; (b) A multi-round dialogue with querying mechanism, consisting of semantic completion, knowledge-guided clarification, and proactive querying processes, to resolve interaction constraints in SQL generation; (c) A data-conditioned SQL generation method selection strategy that supports both an efficient one-step Fine-Tuning approach and a two-step method leveraging Semantic Intermediate Representation for low-cost cross-domain applications. Experiments on both real-world datasets and public benchmarks demonstrate the effectiveness of SiriusBI. User studies further confirm that SiriusBI enhances both productivity and user experience. As an independent service on Tencent's data platform, SiriusBI is deployed across finance, advertising, and cloud sectors, serving dozens of enterprise clients. It achieves over 93% accuracy in SQL generation and reduces data analysts' query time from minutes to seconds in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15961v2">TrainVerify: Equivalence-Based Verification for Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at scale requires parallel execution across thousands of devices, incurring enormous computational costs. Yet, these costly distributed trainings are rarely verified, leaving them prone to silent errors and potentially wasting millions of GPU hours. We introduce TrainVerify, a system for verifiable distributed training of LLMs. Given a deep learning model's logical specification as the ground truth, TrainVerify formally verifies that a distributed parallel execution plan is mathematically equivalent to it. Direct verification is notoriously difficult due to the sheer scale of LLMs which often involves billions of variables and highly intricate computation graphs. Therefore, TrainVerify introduces shape-reduction techniques and a stage-wise parallel verification algorithm that significantly reduces complexity while preserving formal correctness. TrainVerify scales to frontier LLMs, including the successful verification of the Llama3 (405B) and DeepSeek-V3 (671B) training plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19502v1">MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at https://github.com/AlgazinovAleksandr/Multi-Agent-MATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19481v1">LLM-based Multi-Agent System for Intelligent Refactoring of Haskell Code</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 arXiv admin note: text overlap with arXiv:2502.07928
    </div>
    <details class="paper-abstract">
      Refactoring is a constant activity in software development and maintenance. Scale and maintain software systems are based on code refactoring. However, this process is still labor intensive, as it requires programmers to analyze the codebases in detail to avoid introducing new defects. In this research, we put forward a large language model (LLM)-based multi-agent system to automate the refactoring process on Haskell code. The objective of this research is to evaluate the effect of LLM-based agents in performing structured and semantically accurate refactoring on Haskell code. Our proposed multi-agent system based on specialized agents with distinct roles, including code analysis, refactoring execution, verification, and debugging. To test the effectiveness and practical applicability of the multi-agent system, we conducted evaluations using different open-source Haskell codebases. The results of the experiments carried out showed that the proposed LLM-based multi-agent system could average 11.03% decreased complexity in code, an improvement of 22.46% in overall code quality, and increase performance efficiency by an average of 13.27%. Furthermore, memory allocation was optimized by up to 14.57%. These results highlight the ability of LLM-based multi-agent in managing refactoring tasks targeted toward functional programming paradigms. Our findings hint that LLM-based multi-agent systems integration into the refactoring of functional programming languages can enhance maintainability and support automated development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19399v1">Automated Detection of Pre-training Text in Black-box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      Detecting whether a given text is a member of the pre-training data of Large Language Models (LLMs) is crucial for ensuring data privacy and copyright protection. Most existing methods rely on the LLM's hidden information (e.g., model parameters or token probabilities), making them ineffective in the black-box setting, where only input and output texts are accessible. Although some methods have been proposed for the black-box setting, they rely on massive manual efforts such as designing complicated questions or instructions. To address these issues, we propose VeilProbe, the first framework for automatically detecting LLMs' pre-training texts in a black-box setting without human intervention. VeilProbe utilizes a sequence-to-sequence mapping model to infer the latent mapping feature between the input text and the corresponding output suffix generated by the LLM. Then it performs the key token perturbations to obtain more distinguishable membership features. Additionally, considering real-world scenarios where the ground-truth training text samples are limited, a prototype-based membership classifier is introduced to alleviate the overfitting issue. Extensive evaluations on three widely used datasets demonstrate that our framework is effective and superior in the black-box setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18082v2">Statistical Multicriteria Evaluation of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Assessing the quality of LLM-generated text remains a fundamental challenge in natural language processing. Current evaluation approaches often rely on isolated metrics or simplistic aggregations that fail to capture the nuanced trade-offs between coherence, diversity, fluency, and other relevant indicators of text quality. In this work, we adapt a recently proposed framework for statistical inference based on Generalized Stochastic Dominance (GSD) that addresses three critical limitations in existing benchmarking methodologies: the inadequacy of single-metric evaluation, the incompatibility between cardinal automatic metrics and ordinal human judgments, and the lack of inferential statistical guarantees. The GSD-front approach enables simultaneous evaluation across multiple quality dimensions while respecting their different measurement scales, building upon partial orders of decoding strategies, thus avoiding arbitrary weighting of the involved metrics. By applying this framework to evaluate common decoding strategies against human-generated text, we demonstrate its ability to identify statistically significant performance differences while accounting for potential deviations from the i.i.d. assumption of the sampling design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15066v2">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, the functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. To address these challenges, we introduce ChatModel, the first LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 55.02% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Furthermore, it accelerated the iterative process of reference model design and validation by an average of 5.90x compared to traditional approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18631v2">ReDit: Reward Dithering for Improved LLM Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 10 pages, 15 figures
    </div>
    <details class="paper-abstract">
      DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v4">MCP-Zero: Active Tool Discovery for Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      True intelligence requires active capability acquisition, yet current LLM agents inject pre-defined tool schemas into prompts, reducing models to passive selectors and falling short of robust general-purpose agency. We introduce MCP-Zero, an active agent framework that restores tool discovery autonomy to LLMs themselves. Instead of overwhelming models with all available tools, MCP-Zero enables agents to actively identify capability gaps, and request specific tools on-demand, transforming them from large-scale retrievers into genuine autonomous agents. The framework operates through three core mechanisms: (1) Active Tool Request, where models autonomously generate structured requests specifying their exact tool requirements; (2) Hierarchical Semantic Routing, a two-stage algorithm that matches requests to relevant servers and tools through improved semantic alignment; (3) Iterative Capability Extension, enabling agents to progressively build cross-domain toolchains while maintaining minimal context footprint. We construct MCP-tools, a comprehensive dataset of 308 MCP servers and 2,797 tools from the official Model-Context-Protocol repository. Experiments demonstrate that MCP-Zero preserves agent autonomy while achieving substantial efficiency gains: (i) accurate tool selection from nearly 3k candidates across 248.1k tokens; (ii) 98\% reduction in token consumption on APIBank while maintaining high accuracy; and (iii) consistent multi-turn performance that scales with tool ecosystem growth. This work establishes active tool discovery as a fundamental design pattern for scalable autonomous agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13816v3">Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 ACL 2025 main; camera ready
    </div>
    <details class="paper-abstract">
      While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on the knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19290v1">Skywork-SWE: Unveiling Data Scaling Laws for Software Engineering in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Software engineering (SWE) has recently emerged as a crucial testbed for next-generation LLM agents, demanding inherent capabilities in two critical dimensions: sustained iterative problem-solving (e.g., >50 interaction rounds) and long-context dependency resolution (e.g., >32k tokens). However, the data curation process in SWE remains notoriously time-consuming, as it heavily relies on manual annotation for code file filtering and the setup of dedicated runtime environments to execute and validate unit tests. Consequently, most existing datasets are limited to only a few thousand GitHub-sourced instances. To this end, we propose an incremental, automated data-curation pipeline that systematically scales both the volume and diversity of SWE datasets. Our dataset comprises 10,169 real-world Python task instances from 2,531 distinct GitHub repositories, each accompanied by a task specified in natural language and a dedicated runtime-environment image for automated unit-test validation. We have carefully curated over 8,000 successfully runtime-validated training trajectories from our proposed SWE dataset. When fine-tuning the Skywork-SWE model on these trajectories, we uncover a striking data scaling phenomenon: the trained model's performance for software engineering capabilities in LLMs continues to improve as the data size increases, showing no signs of saturation. Notably, our Skywork-SWE model achieves 38.0% pass@1 accuracy on the SWE-bench Verified benchmark without using verifiers or multiple rollouts, establishing a new state-of-the-art (SOTA) among the Qwen2.5-Coder-32B-based LLMs built on the OpenHands agent framework. Furthermore, with the incorporation of test-time scaling techniques, the performance further improves to 47.0% accuracy, surpassing the previous SOTA results for sub-32B parameter models. We release the Skywork-SWE-32B model checkpoint to accelerate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19287v1">Generating and Understanding Tests via Path-Aware Symbolic Execution with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Symbolic execution is a widely used technique for test generation, offering systematic exploration of program paths through constraint solving. However, it is fundamentally constrained by the capability to model the target code including library functions in terms of symbolic constraint and the capability of underlying constraint solvers. As a result, many paths involving complex features remain unanalyzed or insufficiently modeled. Recent advances in large language models (LLMs) have shown promise in generating diverse and valid test inputs. Yet, LLMs lack mechanisms for systematically enumerating program paths and often fail to cover subtle corner cases. We observe that directly prompting an LLM with the full program leads to missed coverage of interesting paths. In this paper, we present PALM, a test generation system that combines symbolic path enumeration with LLM-assisted test generation. PALM statically enumerates possible paths through AST-level analysis and transforms each into an executable variant with embedded assertions that specify the target path. This avoids the need to translate path constraints into SMT formulae, by instead constructing program variants that LLM can interpret. Importantly, PALM is the first to provide an interactive frontend that visualizes path coverage alongside generated tests, assembling tests based on the specific paths they exercise. A user study with 12 participants demonstrates that PALM's frontend helps users better understand path coverage and identify which paths are actually exercised by PALM-generated tests, through verification and visualization of their path profiles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19262v1">What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17539v2">Breaking Single-Tester Limits: Multi-Agent LLMs for Multi-User Feature Testing</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted to International Conference on Software Engineering (ICSE 2026). arXiv admin note: substantial text overlap with arXiv:2504.15474
    </div>
    <details class="paper-abstract">
      The growing dependence on mobile phones and their apps has made multi-user interactive features, like chat calls, live streaming, and video conferencing, indispensable for bridging the gaps in social connectivity caused by physical and situational barriers. However, automating these interactive features for testing is fraught with challenges, owing to their inherent need for timely, dynamic, and collaborative user interactions, which current automated testing methods inadequately address. Inspired by the concept of agents designed to autonomously and collaboratively tackle problems, we propose MAdroid, a novel multi-agent approach powered by the Large Language Models (LLMs) to automate the multi-user interactive task for app feature testing. Specifically, MAdroid employs two functional types of multi-agents: user agents (Operator) and supervisor agents (Coordinator and Observer). Each agent takes a specific role: the Coordinator directs the interactive task; the Operator mimics user interactions on the device; and the Observer monitors and reviews the task automation process. Our evaluation, which included 41 multi-user interactive tasks, demonstrates the effectiveness of our approach, achieving 82.9% of the tasks with 96.8% action similarity, outperforming the ablation studies and state-of-the-art baselines. Additionally, a preliminary investigation underscores MAdroid's practicality by helping identify 11 multi-user interactive bugs during regression app testing, confirming its potential value in real-world software development contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19993v1">CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Recommender systems play a pivotal role in providing relevant content to users. With the rapid development of large language models (LLMs), researchers have begun utilizing LLMs to build more powerful recommender systems. However, existing approaches that focus on aligning LLMs with recommendation tasks do not fully leverage their sequential information processing capabilities, leading to suboptimal performance. In this paper, we propose a novel system called compressed vocabulary expansion (CoVE). In CoVE, each item is assigned a unique ID within the expanded vocabulary. Our framework effectively capitalizes on sequence understanding abilities of LLMs, significantly enhancing their performance on recommendation tasks. Additionally, we compress the embedding layer, making CoVE practical for large-scale industrial applications. The effectiveness and performance of CoVE are demonstrated through comprehensive experiments on multiple recommendation datasets and comparisons with prior works. Our code can be found at https://github.com/HaochenZhang717/CoVE-official-Repo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19992v1">HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19952v1">CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00523v3">Fuzz-Testing Meets LLM-Based Agents: An Automated and Efficient Framework for Jailbreaking Text-To-Image Generation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generative models have revolutionized content creation by transforming textual descriptions into high-quality images. However, these models are vulnerable to jailbreaking attacks, where carefully crafted prompts bypass safety mechanisms to produce unsafe content. While researchers have developed various jailbreak attacks to expose this risk, these methods face significant limitations, including impractical access requirements, easily detectable unnatural prompts, restricted search spaces, and high query demands on the target system. In this paper, we propose JailFuzzer, a novel fuzzing framework driven by large language model (LLM) agents, designed to efficiently generate natural and semantically meaningful jailbreak prompts in a black-box setting. Specifically, JailFuzzer employs fuzz-testing principles with three components: a seed pool for initial and jailbreak prompts, a guided mutation engine for generating meaningful variations, and an oracle function to evaluate jailbreak success. Furthermore, we construct the guided mutation engine and oracle function by LLM-based agents, which further ensures efficiency and adaptability in black-box settings. Extensive experiments demonstrate that JailFuzzer has significant advantages in jailbreaking T2I models. It generates natural and semantically coherent prompts, reducing the likelihood of detection by traditional defenses. Additionally, it achieves a high success rate in jailbreak attacks with minimal query overhead, outperforming existing methods across all key metrics. This study underscores the need for stronger safety mechanisms in generative models and provides a foundation for future research on defending against sophisticated jailbreaking attacks. JailFuzzer is open-source and available at this repository: https://github.com/YingkaiD/JailFuzzer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16065v3">Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01484v2">LLM Watermarking Using Mixtures and Statistical-to-Computational Gaps</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Given a text, can we determine whether it was generated by a large language model (LLM) or by a human? A widely studied approach to this problem is watermarking. We propose an undetectable and elementary watermarking scheme in the closed setting. Also, in the harder open setting, where the adversary has access to most of the model, we propose an unremovable watermarking scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19897v1">Can LLMs Replace Humans During Code Chunking?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential tools in computer science, especially for tasks involving code understanding and generation. However, existing work does not address many of the unique challenges presented by code written for government applications. In particular, government enterprise software is often written in legacy languages like MUMPS or assembly language code (ALC) and the overall token lengths of these systems exceed the context window size for current commercially available LLMs. Additionally, LLMs are primarily trained on modern software languages and have undergone limited testing with legacy languages, making their ability to understand legacy languages unknown and, hence, an area for empirical study. This paper examines the application of LLMs in the modernization of legacy government code written in ALC and MUMPS, addressing the challenges of input limitations. We investigate various code-chunking methods to optimize the generation of summary module comments for legacy code files, evaluating the impact of code-chunking methods on the quality of documentation produced by different LLMs, including GPT-4o, Claude 3 Sonnet, Mixtral, and Llama 3. Our results indicate that LLMs can select partition points closely aligned with human expert partitioning. We also find that chunking approaches have significant impact on downstream tasks such as documentation generation. LLM-created partitions produce comments that are up to 20% more factual and up to 10% more useful than when humans create partitions. Therefore, we conclude that LLMs can be used as suitable replacements for human partitioning of large codebases during LLM-aided modernization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19884v1">MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      As the demand for on-device Large Language Model (LLM) inference grows, energy efficiency has become a major concern, especially for battery-limited mobile devices. Our analysis shows that the memory-bound LLM decode phase dominates energy use, and yet most existing works focus on accelerating the prefill phase, neglecting energy concerns. We introduce Adaptive Energy-Centric Core Selection (AECS) and integrate it into MNN to create the energy-efficient version, MNN-AECS, the first engine-level system solution without requiring root access or OS modifications for energy-efficient LLM decoding. MNN-AECS is designed to reduce LLM decoding energy while keeping decode speed within an acceptable slowdown threshold by dynamically selecting low-power CPU cores. MNN-AECS is evaluated across 5 Android and 2 iOS devices on 5 popular LLMs of various sizes. Compared to original MNN, MNN-AECS cuts down energy use by 23% without slowdown averaged over all 7 devices and 4 datasets. Against other engines, including llama.cpp, executorch, mllm, and MediaPipe, MNN-AECS delivers 39% to 78% energy saving and 12% to 363% speedup on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19846v1">JoyAgents-R1: Joint Evolution Dynamics for Versatile Multi-LLM Agents with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 33 pages, 7 figures, under review
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has emerged as a prominent paradigm for increasingly complex tasks. However, joint evolution across heterogeneous agents remains challenging due to cooperative inefficiency and training instability. In this paper, we propose the joint evolution dynamics for MARL called JoyAgents-R1, which first applies Group Relative Policy Optimization (GRPO) to the joint training of heterogeneous multi-agents. By iteratively refining agents' large language models (LLMs) and memories, the method achieves holistic equilibrium with optimal decision-making and memory capabilities. Specifically, JoyAgents-R1 first implements node-wise Monte Carlo sampling on the behavior of each agent across entire reasoning trajectories to enhance GRPO sampling efficiency while maintaining policy diversity. Then, our marginal benefit-driven selection strategy identifies top-$K$ sampling groups with maximal reward fluctuations, enabling targeted agent model updates that improve training stability and maximize joint benefits through cost-effective parameter adjustments. Meanwhile, JoyAgents-R1 introduces an adaptive memory evolution mechanism that repurposes GRPO rewards as cost-free supervisory signals to eliminate repetitive reasoning and accelerate convergence. Experiments across general and domain-specific scenarios demonstrate that JoyAgents-R1 achieves performance comparable to that of larger LLMs while built on smaller open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19806v1">LLM-Based Social Simulations Require a Boundary</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      This position paper argues that large language model (LLM)-based social simulations should establish clear boundaries to meaningfully contribute to social science research. While LLMs offer promising capabilities for modeling human-like agents compared to traditional agent-based modeling, they face fundamental limitations that constrain their reliability for social pattern discovery. The core issue lies in LLMs' tendency towards an ``average persona'' that lacks sufficient behavioral heterogeneity, a critical requirement for simulating complex social dynamics. We examine three key boundary problems: alignment (simulated behaviors matching real-world patterns), consistency (maintaining coherent agent behavior over time), and robustness (reproducibility under varying conditions). We propose heuristic boundaries for determining when LLM-based simulations can reliably advance social science understanding. We believe that these simulations are more valuable when focusing on (1) collective patterns rather than individual trajectories, (2) agent behaviors aligning with real population averages despite limited variance, and (3) proper validation methods available for testing simulation robustness. We provide a practical checklist to guide researchers in determining the appropriate scope and claims for LLM-based social simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19794v1">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19775v1">Canary in the Mine: An LLM Augmented Survey of Disciplinary Complaints to the Ordre des ingénieurs du Québec (OIQ)</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 22 pages, accepted at the American Society of Engineering Education annual conference 2025, pre-print
    </div>
    <details class="paper-abstract">
      This study uses pre-trained LLMs to conduct thematic analysis to investigate disciplinary incidents involving engineers in Quebec, shedding light on critical gaps in engineering education. Through a comprehensive review of the disciplinary register of the Ordre des ing\'enieurs du Qu\'ebec (OIQ)'s disciplinary register for 2010 to 2024, researchers from engineering education and human resources management in technological development laboratories conducted a thematic analysis of reported incidents to identify patterns, trends, and areas for improvement. The analysis aims to uncover the most common types of disciplinary incidents, underlying causes, and implications for the field in how engineering education addresses (or fails to address) these issues. Our findings identify recurring themes, analyze root causes, and offer recommendations for engineering educators and students to mitigate similar incidents. This research has implications for informing curriculum development, professional development, and performance evaluation, ultimately fostering a culture of professionalism and ethical responsibility in engineering. By providing empirical evidence of disciplinary incidents and their causes, this study contributes to evidence-based practices for engineering education and professional development, enhancing the engineering education community's understanding of professionalism and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20185v2">DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 OSDI 2025
    </div>
    <details class="paper-abstract">
      Quantization of Large Language Models (LLMs) has recently gained popularity, particularly for on-device settings with limited hardware resources. While efficient, quantization inevitably degrades model quality, especially in aggressive low-bit settings such as 3-bit and 4-bit precision. In this paper, we propose DecDEC, an inference scheme that improves the quality of low-bit LLMs while preserving the key benefits of quantization: GPU memory savings and latency reduction. DecDEC stores the residual matrix -- the difference between full-precision and quantized weights -- in CPU, and dynamically fetches the residuals for only a small portion of the weights. This portion corresponds to the salient channels, marked by activation outliers, with the fetched residuals helping to correct quantization errors in these channels. Salient channels are identified dynamically at each decoding step by analyzing the input activations -- this enables adaptation to the dynamic nature of activation distribution, thus maximizing the effectiveness of error compensation. We demonstrate the effectiveness of DecDEC by augmenting state-of-the-art quantization methods. For example, DecDEC reduces the perplexity of a 3-bit Llama-3-8B-Instruct model from 10.15 to 9.12 -- outperforming its 3.5-bit counterpart -- while adding less than 0.0003\% to GPU memory usage and incurring only a 1.7\% inference slowdown on NVIDIA RTX 4050 Mobile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12743v2">"I know myself better, but not really greatly": How Well Can LLMs Detect and Explain LLM-Generated Texts?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Distinguishing between human- and LLM-generated texts is crucial given the risks associated with misuse of LLMs. This paper investigates detection and explanation capabilities of current LLMs across two settings: binary (human vs. LLM-generated) and ternary classification (including an ``undecided'' class). We evaluate 6 close- and open-source LLMs of varying sizes and find that self-detection (LLMs identifying their own outputs) consistently outperforms cross-detection (identifying outputs from other LLMs), though both remain suboptimal. Introducing a ternary classification framework improves both detection accuracy and explanation quality across all models. Through comprehensive quantitative and qualitative analyses using our human-annotated dataset, we identify key explanation failures, primarily reliance on inaccurate features, hallucinations, and flawed reasoning. Our findings underscore the limitations of current LLMs in self-detection and self-explanation, highlighting the need for further research to address overfitting and enhance generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19702v1">LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted at ICDAR 2025
    </div>
    <details class="paper-abstract">
      Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{https://github.com/leitro/Differential-Diagnosis-LoRA}{https://github.com/leitro/Differential-Diagnosis-LoRA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15196v2">HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 27 pages,9 figures
    </div>
    <details class="paper-abstract">
      Heuristic algorithms play a vital role in solving combinatorial optimization (CO) problems, yet traditional designs depend heavily on manual expertise and struggle to generalize across diverse instances. We introduce \textbf{HeurAgenix}, a two-stage hyper-heuristic framework powered by large language models (LLMs) that first evolves heuristics and then selects among them automatically. In the heuristic evolution phase, HeurAgenix leverages an LLM to compare seed heuristic solutions with higher-quality solutions and extract reusable evolution strategies. During problem solving, it dynamically picks the most promising heuristic for each problem state, guided by the LLM's perception ability. For flexibility, this selector can be either a state-of-the-art LLM or a fine-tuned lightweight model with lower inference cost. To mitigate the scarcity of reliable supervision caused by CO complexity, we fine-tune the lightweight heuristic selector with a dual-reward mechanism that jointly exploits singals from selection preferences and state perception, enabling robust selection under noisy annotations. Extensive experiments on canonical benchmarks show that HeurAgenix not only outperforms existing LLM-based hyper-heuristics but also matches or exceeds specialized solvers. Code is available at https://github.com/microsoft/HeurAgenix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19676v1">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence, flexibility, and adaptability, and are rapidly changing human production and lifestyle. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to collectively perform more complex tasks. Under this trend, agent communication is regarded as a foundational pillar of the future AI ecosystem, and many organizations intensively begin to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the recent few months. However, this new field exposes significant security hazard, which can cause severe damage to real-world scenarios. To help researchers to quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we first present a clear definition of agent communication and categorize the entire lifecyle of agent communication into three stages: user-agent interaction, agent-agent communication, and agent-environment communication. Next, for each communication phase, we dissect related protocols and analyze its security risks according to the communication characteristics. Then, we summarize and outlook on the possible defense countermeasures for each risk. Finally, we discuss open issues and future directions in this promising research field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19652v1">Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals.
    </details>
</div>
