# llm - 2025_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21820v1">Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21817v1">Out of Distribution, Out of Luck: How Well Can LLMs Trained on Vulnerability Datasets Detect Top 25 CWE Weaknesses?</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Automated vulnerability detection research has made substantial progress, yet its real-world impact remains limited. Current vulnerability datasets suffer from issues including label inaccuracy rates of 20-71%, extensive duplication, and poor coverage of critical CWE types. These issues create a significant "generalization gap" where models achieve misleading self-testing performance (measured on held-out data from same dataset for training) by exploiting spurious correlations rather than learning true vulnerability patterns. Our analysis reveals that many models experience substantial performance drops of up to 40.6% when evaluated on independent data, sometimes underperforming random guessing. To address these limitations, we present a three-part solution. First, we introduce a manually curated test dataset, BenchVul, covering the MITRE Top 25 Most Dangerous CWEs. Second, we construct a high-quality training dataset, TitanVul, comprising 35,045 functions by aggregating seven public sources and applying deduplication and validation using a novel multi-agent LLM framework. Third, we propose a Realistic Vulnerability Generation (RVG) framework, which synthesizes context-aware vulnerability examples for underrepresented but critical CWE types through simulated development workflows. Our evaluation shows the strengths of each component in closing the generalization gap. First, BenchVul shows the limitations of self-testing: models trained on existing datasets, such as BigVul and PrimeVul, experience performance drops on BenchVul (from 0.776 to 0.519 and from 0.567 to 0.337). Second, training models on TitanVul demonstrates improved generalization, with model performance increasing from 0.584 when evaluated on the same dataset to 0.767 when tested on BenchVul. Third, supplementing TitanVul with RVG-generated data yields further gains, increasing model performance by 14.0% to 0.874.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21815v1">HRIPBench: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 15 pages, 5 figures, 12 tables, a dataset
    </div>
    <details class="paper-abstract">
      Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy is designed to improve their health outcomes and reduce safety risks. Some large language models (LLMs) have demonstrated a decent level of medical knowledge, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HRIPBench, a benchmark designed to evaluate LLM's accuracy and safety risks in harm reduction information provision. The benchmark dataset HRIP-Basic has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration of domain knowledge. Our results indicate that state-of-the-art LLMs still struggle to provide accurate harm reduction information, and sometimes, carry out severe safety risks to PWUD. The use of LLMs in harm reduction contexts should be cautiously constrained to avoid inducing negative health outcomes. WARNING: This paper contains illicit content that potentially induces harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21778v1">AU-LLM: Micro-Expression Action Unit Detection via Enhanced LLM-Based Feature Fusion</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The detection of micro-expression Action Units (AUs) is a formidable challenge in affective computing, pivotal for decoding subtle, involuntary human emotions. While Large Language Models (LLMs) demonstrate profound reasoning abilities, their application to the fine-grained, low-intensity domain of micro-expression AU detection remains unexplored. This paper pioneers this direction by introducing \textbf{AU-LLM}, a novel framework that for the first time uses LLM to detect AUs in micro-expression datasets with subtle intensities and the scarcity of data. We specifically address the critical vision-language semantic gap, the \textbf{Enhanced Fusion Projector (EFP)}. The EFP employs a Multi-Layer Perceptron (MLP) to intelligently fuse mid-level (local texture) and high-level (global semantics) visual features from a specialized 3D-CNN backbone into a single, information-dense token. This compact representation effectively empowers the LLM to perform nuanced reasoning over subtle facial muscle movements.Through extensive evaluations on the benchmark CASME II and SAMM datasets, including stringent Leave-One-Subject-Out (LOSO) and cross-domain protocols, AU-LLM establishes a new state-of-the-art, validating the significant potential and robustness of LLM-based reasoning for micro-expression analysis. The codes are available at https://github.com/ZS-liu-JLU/AU-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10541v2">Multi-Modal Hypergraph Enhanced LLM Learning for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages, 4 figures, submitted to IEEE Transactions on Knowledge and Data Engineering
    </div>
    <details class="paper-abstract">
      The burgeoning presence of Large Language Models (LLM) is propelling the development of personalized recommender systems. Most existing LLM-based methods fail to sufficiently explore the multi-view graph structure correlations inherent in recommendation scenarios. To this end, we propose a novel framework, Hypergraph Enhanced LLM Learning for multimodal Recommendation (HeLLM), designed to equip LLMs with the capability to capture intricate higher-order semantic correlations by fusing graph-level contextual signals with sequence-level behavioral patterns. In the recommender pre-training phase, we design a user hypergraph to uncover shared interest preferences among users and an item hypergraph to capture correlations within multimodal similarities among items. The hypergraph convolution and synergistic contrastive learning mechanism are introduced to enhance the distinguishability of learned representations. In the LLM fine-tuning phase, we inject the learned graph-structured embeddings directly into the LLM's architecture and integrate sequential features capturing each user's chronological behavior. This process enables hypergraphs to leverage graph-structured information as global context, enhancing the LLM's ability to perceive complex relational patterns and integrate multimodal information, while also modeling local temporal dynamics. Extensive experiments demonstrate the superiority of our proposed method over state-of-the-art baselines, confirming the advantages of fusing hypergraph-based context with sequential user behavior in LLMs for recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00378v2">iPanda: An LLM-based Agent for Automated Conformance Testing of Communication Protocols</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Conformance testing is essential for ensuring that protocol implementations comply with their specifications. However, traditional testing approaches involve manually creating numerous test cases and scripts, making the process labor-intensive and inefficient. Recently, Large Language Models (LLMs) have demonstrated impressive text comprehension and code generation abilities, providing promising opportunities for automation. In this paper, we propose iPanda, the first framework that leverages LLMs to automate protocol conformance testing. Given a protocol specification document and its implementation, iPanda first employs a keyword-based method to automatically generate comprehensive test cases. Then, it utilizes retrieval-augmented generation and customized CoT strategy to effectively interpret the implementation and produce executable test programs. To further enhance programs' quality, iPanda incorporates an iterative optimization mechanism to refine generated test scripts interactively. Finally, by executing and analyzing the generated tests, iPanda systematically verifies compliance between implementations and protocol specifications. Comprehensive experiments on various protocols show that iPanda significantly outperforms pure LLM-based approaches, improving the success rate (Pass@1) of test-program generation by factors ranging from 4.675 times to 10.751 times.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21653v1">DGP: A Dual-Granularity Prompting Framework for Fraud Detection with Graph-Enhanced LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Real-world fraud detection applications benefit from graph learning techniques that jointly exploit node features, often rich in textual data, and graph structural information. Recently, Graph-Enhanced LLMs emerge as a promising graph learning approach that converts graph information into prompts, exploiting LLMs' ability to reason over both textual and structural information. Among them, text-only prompting, which converts graph information to prompts consisting solely of text tokens, offers a solution that relies only on LLM tuning without requiring additional graph-specific encoders. However, text-only prompting struggles on heterogeneous fraud-detection graphs: multi-hop relations expand exponentially with each additional hop, leading to rapidly growing neighborhoods associated with dense textual information. These neighborhoods may overwhelm the model with long, irrelevant content in the prompt and suppress key signals from the target node, thereby degrading performance. To address this challenge, we propose Dual Granularity Prompting (DGP), which mitigates information overload by preserving fine-grained textual details for the target node while summarizing neighbor information into coarse-grained text prompts. DGP introduces tailored summarization strategies for different data modalities, bi-level semantic abstraction for textual fields and statistical aggregation for numerical features, enabling effective compression of verbose neighbor content into concise, informative prompts. Experiments across public and industrial datasets demonstrate that DGP operates within a manageable token budget while improving fraud detection performance by up to 6.8% (AUPRC) over state-of-the-art methods, showing the potential of Graph-Enhanced LLMs for fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21636v1">StaffPro: an LLM Agent for Joint Staffing and Profiling</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents integrate pre-trained LLMs with modular algorithmic components and have shown remarkable reasoning and decision-making abilities. In this work, we investigate their use for two tightly intertwined challenges in workforce management: staffing, i.e., the assignment and scheduling of tasks to workers, which may require team formation; and profiling, i.e., the continuous estimation of workers' skills, preferences, and other latent attributes from unstructured data. We cast these problems in a formal mathematical framework that links scheduling decisions to latent feature estimation, and we introduce StaffPro, an LLM agent that addresses staffing and profiling jointly. Differently from existing staffing solutions, StaffPro allows expressing optimization objectives using natural language, accepts textual task descriptions and provides high flexibility. StaffPro interacts directly with humans by establishing a continuous human-agent feedback loop, ensuring natural and intuitive use. By analyzing human feedback, our agent continuously estimates the latent features of workers, realizing life-long worker profiling and ensuring optimal staffing performance over time. A consulting firm simulation example demonstrates that StaffPro successfully estimates workers' attributes and generates high quality schedules. With its innovative design, StaffPro offers a robust, interpretable, and human-centric solution for automated personnel management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10635v3">Strategist: Self-improvement of LLM Decision Making via Bi-Level Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 website: https://llm-strategist.github.io
    </div>
    <details class="paper-abstract">
      Traditional reinforcement learning and planning typically requires vast amounts of data and training to develop effective policies. In contrast, large language models (LLMs) exhibit strong generalization and zero-shot capabilities, but struggle with tasks that require detailed planning and decision-making in complex action spaces. We introduce STRATEGIST, a novel approach that integrates the strengths of both methods. Our approach leverages LLMs to search and update high-level strategies (as text), which are then refined and executed by low-level Monte Carlo Tree Search (MCTS). STRATEGIST is a generalizable framework to optimize the strategy through population-based self-play simulations without the need for any training data. We demonstrate the effectiveness of STRATEGIST in learning optimal strategies for competitive, multi-turn games with partial information, including Game of Pure Strategy (GOPS) and multi-agent, hidden-identity discussion games like The Resistance: Avalon. Our results show that agents equipped with STRATEGIST outperform those trained with traditional RL methods, other LLM-based skill acquisition techniques, pre-existing LLM agents across both game environments and achieves comparable performance against human players.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15549v3">Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Code at https://github.com/aengusl/latent-adversarial-training. Models at https://huggingface.co/LLM-LAT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07237v2">Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance across a range of natural language processing tasks. However, their vast number of parameters introduces significant memory challenges during training, particularly when using memory-intensive optimizers like Adam. Existing memory-efficient algorithms often rely on techniques such as singular value decomposition projection or weight freezing. While these approaches help alleviate memory constraints, they generally produce suboptimal results compared to full-rank updates. In this paper, we investigate the memory-efficient method beyond low-rank training, proposing a novel solution called Gradient Wavelet Transform (GWT), which applies wavelet transforms to gradients in order to significantly reduce the memory requirements for maintaining optimizer states. We demonstrate that GWT can be seamlessly integrated with memory-intensive optimizers, enabling efficient training without sacrificing performance. Through extensive experiments on both pre-training and fine-tuning tasks, we show that GWT achieves state-of-the-art performance compared with advanced memory-efficient optimizers and full-rank approaches in terms of both memory usage and training performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03248v2">AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have enabled the creation of multi-modal LLMs that exhibit strong comprehension of visual data such as images and videos. However, these models usually rely on extensive visual tokens from visual encoders, leading to high computational demands, which limits their applicability in resource-constrained environments and for long-context tasks. In this work, we propose a training-free adaptive inference method for multi-modal LLMs that can accommodate a broad range of efficiency requirements with a minimum performance drop. Our method consists of a) iterative token merging based on embedding similarity before LLMs, and b) progressive token pruning within LLM layers based on multi-modal importance. With a minimalist design, our method can be applied to both video and image LLMs. Extensive experiments on diverse video and image benchmarks demonstrate that our method substantially reduces computation load (e.g., a $\textbf{7-fold}$ reduction in FLOPs) while preserving the performance of video and image LLMs. Further, at a similar computational cost, our method outperforms the state-of-the-art methods in long video understanding (e.g., $\textbf{+4.6}$ on MLVU). Additionally, our in-depth analysis provides insights into token redundancy and LLM layer behaviors, offering guidance for future research in designing efficient multi-modal LLMs. Our code is available at https://github.com/LaVi-Lab/AIM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21563v1">Enhancing Graph-based Recommendations with Majority-Voting LLM-Rerank Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Recommendation systems often suffer from data sparsity caused by limited user-item interactions, which degrade their performance and amplify popularity bias in real-world scenarios. This paper proposes a novel data augmentation framework that leverages Large Language Models (LLMs) and item textual descriptions to enrich interaction data. By few-shot prompting LLMs multiple times to rerank items and aggregating the results via majority voting, we generate high-confidence synthetic user-item interactions, supported by theoretical guarantees based on the concentration of measure. To effectively leverage the augmented data in the context of a graph recommendation system, we integrate it into a graph contrastive learning framework to mitigate distributional shift and alleviate popularity bias. Extensive experiments show that our method improves accuracy and reduces popularity bias, outperforming strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v2">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19537v2">Mind the Language Gap in Digital Humanities: LLM-Aided Translation of SKOS Thesauri</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We introduce WOKIE, an open-source, modular, and ready-to-use pipeline for the automated translation of SKOS thesauri. This work addresses a critical need in the Digital Humanities (DH), where language diversity can limit access, reuse, and semantic interoperability of knowledge resources. WOKIE combines external translation services with targeted refinement using Large Language Models (LLMs), balancing translation quality, scalability, and cost. Designed to run on everyday hardware and be easily extended, the application requires no prior expertise in machine translation or LLMs. We evaluate WOKIE across several DH thesauri in 15 languages with different parameters, translation services and LLMs, systematically analysing translation quality, performance, and ontology matching improvements. Our results show that WOKIE is suitable to enhance the accessibility, reuse, and cross-lingual interoperability of thesauri by hurdle-free automated translation and improved ontology matching performance, supporting more inclusive and multilingual research infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21538v1">Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21507v1">VAGU & GtS: LLM-Based Benchmark and Framework for Joint Video Anomaly Grounding and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 21 pages, 19 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Video Anomaly Detection (VAD) aims to identify anomalous events in videos and accurately determine their time intervals. Current VAD methods mainly fall into two categories: traditional DNN-based approaches that focus on temporal localization, and LLM-based approaches that emphasize semantic understanding. Both anomaly understanding and grounding are essential for comprehensive video anomaly detection and can complement each other. However, no existing model or dataset supports both tasks simultaneously. To address this, we introduce VAGU (Video Anomaly Grounding and Understanding), the first benchmark to integrate both tasks. Each VAGU instance includes annotations for anomaly category, semantic explanation, precise temporal grounding and Video QA. We also provide multiple-choice Video QA for objective evaluation. Based on this dataset, we propose Glance then Scrutinize (GtS), a training-free framework guided by textual prompts. The framework first enables coarse localization of high-probability anomalous regions, followed by detailed anomaly interpretation and temporal boundary refinement. Additionally, we propose the JeAUG metric, which jointly evaluates semantic interpretability and temporal precision, overcoming the limitations of traditional metrics. Extensive experiments verify the effectiveness of our benchmark, framework, and evaluation metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14479v4">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21504v1">Evaluation and Benchmarking of LLM Agents: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The rise of LLM-based agents has opened new frontiers in AI applications, yet evaluating these agents remains a complex and underdeveloped area. This survey provides an in-depth overview of the emerging field of LLM agent evaluation, introducing a two-dimensional taxonomy that organizes existing work along (1) evaluation objectives -- what to evaluate, such as agent behavior, capabilities, reliability, and safety -- and (2) evaluation process -- how to evaluate, including interaction modes, datasets and benchmarks, metric computation methods, and tooling. In addition to taxonomy, we highlight enterprise-specific challenges, such as role-based access to data, the need for reliability guarantees, dynamic and long-horizon interactions, and compliance, which are often overlooked in current research. We also identify future research directions, including holistic, more realistic, and scalable evaluation. This work aims to bring clarity to the fragmented landscape of agent evaluation and provide a framework for systematic assessment, enabling researchers and practitioners to evaluate LLM agents for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21485v1">HLSDebugger: Identification and Correction of Logic Bugs in HLS Code with LLM Solutions</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 This work has been accepted at ICCAD 2025 (International Conference on Computer-Aided Design)
    </div>
    <details class="paper-abstract">
      High-level synthesis (HLS) accelerates hardware design by enabling the automatic translation of high-level descriptions into efficient hardware implementations. However, debugging HLS code is a challenging and labor-intensive task, especially for novice circuit designers or software engineers without sufficient hardware domain knowledge. The recent emergence of Large Language Models (LLMs) is promising in automating the HLS debugging process. Despite the great potential, three key challenges persist when applying LLMs to HLS logic debugging: 1) High-quality circuit data for training LLMs is scarce, posing a significant challenge. 2) Debugging logic bugs in hardware is inherently more complex than identifying software bugs with existing golden test cases. 3) The absence of reliable test cases requires multi-tasking solutions, performing both bug identification and correction. complicates the multi-tasking required for effective HLS debugging. In this work, we propose a customized solution named HLSDebugger to address the challenges. HLSDebugger first generates and releases a large labeled dataset with 300K data samples, targeting HLS logic bugs. The HLSDebugger model adopts an encoder-decoder structure, performing bug location identification, bug type prediction, and bug correction with the same model. HLSDebugger significantly outperforms advanced LLMs like GPT-4 in bug identification and by more than 3x in bug correction. It makes a substantial advancement in the exploration of automated debugging of HLS code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21482v1">Improving Task Diversity in Label Efficient Supervised Finetuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but developing high-performing models for specialized applications often requires substantial human annotation -- a process that is time-consuming, labor-intensive, and expensive. In this paper, we address the label-efficient learning problem for supervised finetuning (SFT) by leveraging task-diversity as a fundamental principle for effective data selection. This is markedly different from existing methods based on the prompt-diversity. Our approach is based on two key observations: 1) task labels for different prompts are often readily available; 2) pre-trained models have significantly varying levels of confidence across tasks. We combine these facts to devise a simple yet effective sampling strategy: we select examples across tasks using an inverse confidence weighting strategy. This produces models comparable to or better than those trained with more complex sampling procedures, while being significantly easier to implement and less computationally intensive. Notably, our experimental results demonstrate that this method can achieve better accuracy than training on the complete dataset (a 4\% increase in MMLU score). Across various annotation budgets and two instruction finetuning datasets, our algorithm consistently performs at or above the level of the best existing methods, while reducing annotation costs by up to 80\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21476v1">Which LLMs Get the Joke? Probing Non-STEM Reasoning Abilities with HumorBench</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      We present HumorBench, a benchmark designed to evaluate large language models' (LLMs) ability to reason about and explain sophisticated humor in cartoon captions. As reasoning models increasingly saturate existing benchmarks in mathematics and science, novel and challenging evaluations of model intelligence beyond STEM domains are essential. Reasoning is fundamentally involved in text-based humor comprehension, requiring the identification of connections between concepts in cartoons/captions and external cultural references, wordplays, and other mechanisms. HumorBench includes approximately 300 unique cartoon-caption pairs from the New Yorker Caption Contest and Cartoonstock.com, with expert-annotated evaluation rubrics identifying essential joke elements. LLMs are evaluated based on their explanations towards the humor and abilities in identifying the joke elements. To perform well on this task, models must form and test hypotheses about associations between concepts, potentially backtracking from initial interpretations to arrive at the most plausible explanation. Our extensive benchmarking of current SOTA models reveals three key insights: (1) LLM progress on STEM reasoning transfers effectively to humor comprehension; (2) models trained exclusively on STEM reasoning data still perform well on HumorBench, demonstrating strong transferability of reasoning abilities; and (3) test-time scaling by increasing thinking token budgets yields mixed results across different models in humor reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21471v1">An LLM Driven Agent Framework for Automated Infrared Spectral Multi Task Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Infrared spectroscopy offers rapid, non destructive measurement of chemical and material properties but suffers from high dimensional, overlapping spectral bands that challenge conventional chemometric approaches. Emerging large language models (LLMs), with their capacity for generalization and reasoning, offer promising potential for automating complex scientific workflows. Despite this promise, their application in IR spectral analysis remains largely unexplored. This study addresses the critical challenge of achieving accurate, automated infrared spectral interpretation under low-data conditions using an LLM-driven framework. We introduce an end-to-end, large language model driven agent framework that integrates a structured literature knowledge base, automated spectral preprocessing, feature extraction, and multi task reasoning in a unified pipeline. By querying a curated corpus of peer reviewed IR publications, the agent selects scientifically validated routines. The selected methods transform each spectrum into low dimensional feature sets, which are fed into few shot prompt templates for classification, regression, and anomaly detection. A closed loop, multi turn protocol iteratively appends mispredicted samples to the prompt, enabling dynamic refinement of predictions. Across diverse materials: stamp pad ink, Chinese medicine, Pu'er tea, Citri Reticulatae Pericarpium and waste water COD datasets, the multi turn LLM consistently outperforms single turn inference, rivaling or exceeding machine learning and deep learning models under low data regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v3">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21447v1">LLM4VV: Evaluating Cutting-Edge LLMs for Generation and Evaluation of Directive-Based Parallel Programming Model Compiler Tests</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      The usage of Large Language Models (LLMs) for software and test development has continued to increase since LLMs were first introduced, but only recently have the expectations of LLMs become more realistic. Verifying the correctness of code generated by LLMs is key to improving their usefulness, but there have been no comprehensive and fully autonomous solutions developed yet. Hallucinations are a major concern when LLMs are applied blindly to problems without taking the time and effort to verify their outputs, and an inability to explain the logical reasoning of LLMs leads to issues with trusting their results. To address these challenges while also aiming to effectively apply LLMs, this paper proposes a dual-LLM system (i.e. a generative LLM and a discriminative LLM) and experiments with the usage of LLMs for the generation of a large volume of compiler tests. We experimented with a number of LLMs possessing varying parameter counts and presented results using ten carefully-chosen metrics that we describe in detail in our narrative. Through our findings, it is evident that LLMs possess the promising potential to generate quality compiler tests and verify them automatically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v2">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03885v5">InfiniteHBD: Building Datacenter-Scale High-Bandwidth Domain for LLM with Optical Circuit Switching Transceivers</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Scaling Large Language Model (LLM) training relies on multi-dimensional parallelism, where High-Bandwidth Domains (HBDs) are critical for communication-intensive parallelism like Tensor Parallelism (TP) and Expert Parallelism (EP). However, existing HBD architectures face fundamental limitations in scalability, cost, and fault resiliency: switch-centric HBDs (e.g., NVL-72) incur prohibitive scaling costs, while GPU-centric HBDs (e.g., TPUv3/Dojo) suffer from severe fault propagation. Switch-GPU hybrid HBDs such as TPUv4 take a middle-ground approach, but the fault explosion radius remains large at the cube level (e.g., 64 TPUs). We propose InfiniteHBD, a novel transceiver-centric HBD architecture that unifies connectivity and dynamic switching at the transceiver level using Optical Circuit Switching (OCS). By embedding OCS within each transceiver, InfiniteHBD achieves reconfigurable point-to-multipoint connectivity, allowing the topology to adapt to variable-size rings. This design provides: i) datacenter-wide scalability without cost explosion; ii) fault resilience by isolating failures to a single node, and iii) full bandwidth utilization for fault-free GPUs. Key innovations include a Silicon Photonic (SiPh)-based low-cost OCS transceiver (OCSTrx), a reconfigurable k-hop ring topology co-designed with intra-/inter-node communication, and an HBD-DCN orchestration algorithm maximizing GPU utilization while minimizing cross-ToR datacenter network traffic. The evaluation demonstrates that InfiniteHBD achieves 31% of the cost of NVL-72, near-zero GPU waste ratio (over one order of magnitude lower than NVL-72 and TPUv4), near-zero cross-ToR traffic when node fault ratios are under 7%, and improves Model FLOPs Utilization by 3.37x compared to NVIDIA DGX (8 GPUs per Node).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12225v2">Mining Intrinsic Rewards from LLM Hidden States for Efficient Best-of-N Sampling</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Enhancing Large Language Model (LLM)'s performance with best-of-N sampling is effective and has attracted significant attention. However, it is computationally prohibitive due to massive, data-hungry text-based reward models. By changing the data source from text to hidden states, we introduce SWIFT (Simple Weighted Intrinsic Feedback Technique), a novel, lightweight technique that leverages the rich information embedded in LLM hidden states to address these issues, which operates on token-level and consists of only linear layers. Extensive experiments show that SWIFT outperforms baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training, demonstrating significant efficiency improvement. SWIFT's robust scalability, applicability to some closed-source models via logits, and ability to be combined with traditional reward models to yield further performance gains underscore its practical value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21428v1">MemTool: Optimizing Short-Term Memory Management for Dynamic Tool Calling in LLM Agent Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 23 Pages, 20 Figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have shown significant autonomous capabilities in dynamically searching and incorporating relevant tools or Model Context Protocol (MCP) servers for individual queries. However, fixed context windows limit effectiveness in multi-turn interactions requiring repeated, independent tool usage. We introduce MemTool, a short-term memory framework enabling LLM agents to dynamically manage tools or MCP server contexts across multi-turn conversations. MemTool offers three agentic architectures: 1) Autonomous Agent Mode, granting full tool management autonomy, 2) Workflow Mode, providing deterministic control without autonomy, and 3) Hybrid Mode, combining autonomous and deterministic control. Evaluating each MemTool mode across 13+ LLMs on the ScaleMCP benchmark, we conducted experiments over 100 consecutive user interactions, measuring tool removal ratios (short-term memory efficiency) and task completion accuracy. In Autonomous Agent Mode, reasoning LLMs achieve high tool-removal efficiency (90-94% over a 3-window average), while medium-sized models exhibit significantly lower efficiency (0-60%). Workflow and Hybrid modes consistently manage tool removal effectively, whereas Autonomous and Hybrid modes excel at task completion. We present trade-offs and recommendations for each MemTool mode based on task accuracy, agency, and model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22289v1">Intent Recognition and Out-of-Scope Detection using LLMs in Multi-party Conversations</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 Accepted for publication at SIGDIAL 2025
    </div>
    <details class="paper-abstract">
      Intent recognition is a fundamental component in task-oriented dialogue systems (TODS). Determining user intents and detecting whether an intent is Out-of-Scope (OOS) is crucial for TODS to provide reliable responses. However, traditional TODS require large amount of annotated data. In this work we propose a hybrid approach to combine BERT and LLMs in zero and few-shot settings to recognize intents and detect OOS utterances. Our approach leverages LLMs generalization power and BERT's computational efficiency in such scenarios. We evaluate our method on multi-party conversation corpora and observe that sharing information from BERT outputs to LLMs leads to system performance improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22286v1">Meaning-infused grammar: Gradient Acceptability Shapes the Geometric Representations of Constructions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
      | 💬 5 pages, 3 figures, Accepted for publication at the Second International Workshop on Construction Grammars and NLP at the 16th International Conference for Computational Semantics (IWCS) 2025
    </div>
    <details class="paper-abstract">
      The usage-based constructionist (UCx) approach posits that language comprises a network of learned form-meaning pairings (constructions) whose use is largely determined by their meanings or functions, requiring them to be graded and probabilistic. This study investigates whether the internal representations in Large Language Models (LLMs) reflect the proposed function-infused gradience. We analyze the neural representations of the English dative constructions (Double Object and Prepositional Object) in Pythia-$1.4$B, using a dataset of $5000$ sentence pairs systematically varied for human-rated preference strength. A macro-level geometric analysis finds that the separability between construction representations, as measured by Energy Distance or Jensen-Shannon Divergence, is systematically modulated by gradient preference strength. More prototypical exemplars of each construction occupy more distinct regions in the activation space of LLMs. These results provide strong evidence that LLMs learn rich, meaning-infused, graded representations of constructions and offer support for geometric measures of basic constructionist principles in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22267v1">Promoting Online Safety by Simulating Unsafe Conversations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Generative AI, including large language models (LLMs) have the potential -- and already are being used -- to increase the speed, scale, and types of unsafe conversations online. LLMs lower the barrier for entry for bad actors to create unsafe conversations in particular because of their ability to generate persuasive and human-like text. In our current work, we explore ways to promote online safety by teaching people about unsafe conversations that can occur online with and without LLMs. We build on prior work that shows that LLMs can successfully simulate scam conversations. We also leverage research in the learning sciences that shows that providing feedback on one's hypothetical actions can promote learning. In particular, we focus on simulating scam conversations using LLMs. Our work incorporates two LLMs that converse with each other to simulate realistic, unsafe conversations that people may encounter online between a scammer LLM and a target LLM but users of our system are asked provide feedback to the target LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14907v2">GneissWeb: Preparing High Quality Data for LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-07-29
    </div>
    <details class="paper-abstract">
      Data quantity and quality play a vital role in determining the performance of Large Language Models (LLMs). High-quality data, in particular, can significantly boost the LLM's ability to generalize on a wide range of downstream tasks. Large pre-training datasets for leading LLMs remain inaccessible to the public, whereas many open datasets are small in size (less than 5 trillion tokens), limiting their suitability for training large models. In this paper, we introduce GneissWeb, a large dataset yielding around 10 trillion tokens that caters to the data quality and quantity requirements of training LLMs. Our GneissWeb recipe that produced the dataset consists of sharded exact sub-string deduplication and a judiciously constructed ensemble of quality filters. GneissWeb achieves a favorable trade-off between data quality and quantity, producing models that outperform models trained on state-of-the-art open large datasets (5+ trillion tokens). We show that models trained using GneissWeb dataset outperform those trained on FineWeb-V1.1.0 by 2.73 percentage points in terms of average score computed on a set of 11 commonly used benchmarks (both zero-shot and few-shot) for pre-training dataset evaluation. When the evaluation set is extended to 20 benchmarks (both zero-shot and few-shot), models trained using GneissWeb still achieve a 1.75 percentage points advantage over those trained on FineWeb-V1.1.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21028v1">Multi-Agent-as-Judge: Aligning LLM-Agent-Based Automated Evaluation with Multi-Dimensional Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Nearly all human work is collaborative; thus, the evaluation of real-world NLP applications often requires multiple dimensions that align with diverse human perspectives. As real human evaluator resources are often scarce and costly, the emerging "LLM-as-a-judge" paradigm sheds light on a promising approach to leverage LLM agents to believably simulate human evaluators. Yet, to date, existing LLM-as-a-judge approaches face two limitations: persona descriptions of agents are often arbitrarily designed, and the frameworks are not generalizable to other tasks. To address these challenges, we propose MAJ-EVAL, a Multi-Agent-as-Judge evaluation framework that can automatically construct multiple evaluator personas with distinct dimensions from relevant text documents (e.g., research papers), instantiate LLM agents with the personas, and engage in-group debates with multi-agents to Generate multi-dimensional feedback. Our evaluation experiments in both the educational and medical domains demonstrate that MAJ-EVAL can generate evaluation results that better align with human experts' ratings compared with conventional automated evaluation metrics and existing LLM-as-a-judge methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12854v3">Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21017v1">MIRAGE-Bench: LLM Agent is Hallucinating and Where to Find Them</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Code and data: https://github.com/sunblaze-ucb/mirage-bench.git
    </div>
    <details class="paper-abstract">
      Hallucinations pose critical risks for large language model (LLM)-based agents, often manifesting as hallucinative actions resulting from fabricated or misinterpreted information within the cognitive context. While recent studies have exposed such failures, existing evaluations remain fragmented and lack a principled testbed. In this paper, we present MIRAGE-Bench--Measuring Illusions in Risky AGEnt settings--the first unified benchmark for eliciting and evaluating hallucinations in interactive LLM-agent scenarios. We begin by introducing a three-part taxonomy to address agentic hallucinations: actions that are unfaithful to (i) task instructions, (ii) execution history, or (iii) environment observations. To analyze, we first elicit such failures by performing a systematic audit of existing agent benchmarks, then synthesize test cases using a snapshot strategy that isolates decision points in deterministic and reproducible manners. To evaluate hallucination behaviors, we adopt a fine-grained-level LLM-as-a-Judge paradigm with tailored risk-aware prompts, enabling scalable, high-fidelity assessment of agent actions without enumerating full action spaces. MIRAGE-Bench provides actionable insights on failure modes of LLM agents and lays the groundwork for principled progress in mitigating hallucinations in interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02087v2">Evaluating the Promise and Pitfalls of LLMs in Hiring Decisions</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages, 2 figures, 2 tables. Submitted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) in hiring promises to streamline candidate screening, but it also raises serious concerns regarding accuracy and algorithmic bias where sufficient safeguards are not in place. In this work, we benchmark several state-of-the-art foundational LLMs - including models from OpenAI, Anthropic, Google, Meta, and Deepseek, and compare them with our proprietary domain-specific hiring model (Match Score) for job candidate matching. We evaluate each model's predictive accuracy (ROC AUC, Precision-Recall AUC, F1-score) and fairness (impact ratio of cut-off analysis across declared gender, race, and intersectional subgroups). Our experiments on a dataset of roughly 10,000 real-world recent candidate-job pairs show that Match Score outperforms the general-purpose LLMs on accuracy (ROC AUC 0.85 vs 0.77) and achieves significantly more equitable outcomes across demographic groups. Notably, Match Score attains a minimum race-wise impact ratio of 0.957 (near-parity), versus 0.809 or lower for the best LLMs, (0.906 vs 0.773 for the intersectionals, respectively). We discuss why pretraining biases may cause LLMs with insufficient safeguards to propagate societal biases in hiring scenarios, whereas a bespoke supervised model can more effectively mitigate these biases. Our findings highlight the importance of domain-specific modeling and bias auditing when deploying AI in high-stakes domains such as hiring, and caution against relying on off-the-shelf LLMs for such tasks without extensive fairness safeguards. Furthermore, we show with empirical evidence that there shouldn't be a dichotomy between choosing accuracy and fairness in hiring: a well-designed algorithm can achieve both accuracy in hiring and fairness in outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v1">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20977v1">Repairing vulnerabilities without invisible hands. A differentiated replication study on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Background: Automated Vulnerability Repair (AVR) is a fast-growing branch of program repair. Recent studies show that large language models (LLMs) outperform traditional techniques, extending their success beyond code generation and fault detection. Hypothesis: These gains may be driven by hidden factors -- "invisible hands" such as training-data leakage or perfect fault localization -- that let an LLM reproduce human-authored fixes for the same code. Objective: We replicate prior AVR studies under controlled conditions by deliberately adding errors to the reported vulnerability location in the prompt. If LLMs merely regurgitate memorized fixes, both small and large localization errors should yield the same number of correct patches, because any offset should divert the model from the original fix. Method: Our pipeline repairs vulnerabilities from the Vul4J and VJTrans benchmarks after shifting the fault location by n lines from the ground truth. A first LLM generates a patch, a second LLM reviews it, and we validate the result with regression and proof-of-vulnerability tests. Finally, we manually audit a sample of patches and estimate the error rate with the Agresti-Coull-Wilson method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17137v2">Cog-TiPRO: Iterative Prompt Refinement with LLMs to Detect Cognitive Decline via Longitudinal Voice Assistant Commands</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 IEEE Global Communications Conference (GlobeCom) 2025
    </div>
    <details class="paper-abstract">
      Early detection of cognitive decline is crucial for enabling interventions that can slow neurodegenerative disease progression. Traditional diagnostic approaches rely on labor-intensive clinical assessments, which are impractical for frequent monitoring. Our pilot study investigates voice assistant systems (VAS) as non-invasive tools for detecting cognitive decline through longitudinal analysis of speech patterns in voice commands. Over an 18-month period, we collected voice commands from 35 older adults, with 15 participants providing daily at-home VAS interactions. To address the challenges of analyzing these short, unstructured and noisy commands, we propose Cog-TiPRO, a framework that combines (1) LLM-driven iterative prompt refinement for linguistic feature extraction, (2) HuBERT-based acoustic feature extraction, and (3) transformer-based temporal modeling. Using iTransformer, our approach achieves 73.80% accuracy and 72.67% F1-score in detecting MCI, outperforming its baseline by 27.13%. Through our LLM approach, we identify linguistic features that uniquely characterize everyday command usage patterns in individuals experiencing cognitive decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20957v1">Your AI, Not Your View: The Bias of LLMs in Investment Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      In finance, Large Language Models (LLMs) face frequent knowledge conflicts due to discrepancies between pre-trained parametric knowledge and real-time market data. These conflicts become particularly problematic when LLMs are deployed in real-world investment services, where misalignment between a model's embedded preferences and those of the financial institution can lead to unreliable recommendations. Yet little research has examined what investment views LLMs actually hold. We propose an experimental framework to investigate such conflicts, offering the first quantitative analysis of confirmation bias in LLM-based investment analysis. Using hypothetical scenarios with balanced and imbalanced arguments, we extract models' latent preferences and measure their persistence. Focusing on sector, size, and momentum, our analysis reveals distinct, model-specific tendencies. In particular, we observe a consistent preference for large-cap stocks and contrarian strategies across most models. These preferences often harden into confirmation bias, with models clinging to initial judgments despite counter-evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16117v2">BDIViz: An Interactive Visualization System for Biomedical Schema Matching with LLM-Powered Validation</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 9 figures. Accepted to IEEE VIS 2025 (Full Papers Track, submission ID 1204)
    </div>
    <details class="paper-abstract">
      Biomedical data harmonization is essential for enabling exploratory analyses and meta-studies, but the process of schema matching - identifying semantic correspondences between elements of disparate datasets (schemas) - remains a labor-intensive and error-prone task. Even state-of-the-art automated methods often yield low accuracy when applied to biomedical schemas due to the large number of attributes and nuanced semantic differences between them. We present BDIViz, a novel visual analytics system designed to streamline the schema matching process for biomedical data. Through formative studies with domain experts, we identified key requirements for an effective solution and developed interactive visualization techniques that address both scalability challenges and semantic ambiguity. BDIViz employs an ensemble approach that combines multiple matching methods with LLM-based validation, summarizes matches through interactive heatmaps, and provides coordinated views that enable users to quickly compare attributes and their values. Our method-agnostic design allows the system to integrate various schema matching algorithms and adapt to application-specific needs. Through two biomedical case studies and a within-subject user study with domain experts, we demonstrate that BDIViz significantly improves matching accuracy while reducing cognitive load and curation time compared to baseline approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20870v1">A Human-in-the-loop Approach to Robot Action Replanning through LLM Common-Sense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      To facilitate the wider adoption of robotics, accessible programming tools are required for non-experts. Observational learning enables intuitive human skills transfer through hands-on demonstrations, but relying solely on visual input can be inefficient in terms of scalability and failure mitigation, especially when based on a single demonstration. This paper presents a human-in-the-loop method for enhancing the robot execution plan, automatically generated based on a single RGB video, with natural language input to a Large Language Model (LLM). By including user-specified goals or critical task aspects and exploiting the LLM common-sense reasoning, the system adjusts the vision-based plan to prevent potential failures and adapts it based on the received instructions. Experiments demonstrated the framework intuitiveness and effectiveness in correcting vision-derived errors and adapting plans without requiring additional demonstrations. Moreover, interactive plan refinement and hallucination corrections promoted system robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20849v1">Latent Inter-User Difference Modeling for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into users' daily lives, leading to a growing demand for personalized outputs. Previous work focuses on leveraging a user's own history, overlooking inter-user differences that are crucial for effective personalization. While recent work has attempted to model such differences, the reliance on language-based prompts often hampers the effective extraction of meaningful distinctions. To address these issues, we propose Difference-aware Embedding-based Personalization (DEP), a framework that models inter-user differences in the latent space instead of relying on language prompts. DEP constructs soft prompts by contrasting a user's embedding with those of peers who engaged with similar content, highlighting relative behavioral signals. A sparse autoencoder then filters and compresses both user-specific and difference-aware embeddings, preserving only task-relevant features before injecting them into a frozen LLM. Experiments on personalized review generation show that DEP consistently outperforms baseline methods across multiple metrics. Our code is available at https://github.com/SnowCharmQ/DEP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00582v2">Do Language Models Mirror Human Confidence? Exploring Psychological Insights to Address Overconfidence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by ACL 2025 Findings, 20 pages
    </div>
    <details class="paper-abstract">
      Psychology research has shown that humans are poor at estimating their performance on tasks, tending towards underconfidence on easy tasks and overconfidence on difficult tasks. We examine three LLMs, Llama-3-70B-instruct, Claude-3-Sonnet, and GPT-4o, on a range of QA tasks of varying difficulty, and show that models exhibit subtle differences from human patterns of overconfidence: less sensitive to task difficulty, and when prompted to answer based on different personas -- e.g., expert vs layman, or different race, gender, and ages -- the models will respond with stereotypically biased confidence estimations even though their underlying answer accuracy remains the same. Based on these observations, we propose Answer-Free Confidence Estimation (AFCE) to improve confidence calibration and LLM interpretability in these settings. AFCE is a self-assessment method that employs two stages of prompting, first eliciting only confidence scores on questions, then asking separately for the answer. Experiments on the MMLU and GPQA datasets spanning subjects and difficulty show that this separation of tasks significantly reduces overconfidence and delivers more human-like sensitivity to task difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20774v1">evalSmarT: An LLM-Based Framework for Evaluating Smart Contract Generated Comments</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 4 pages, 4 tables
    </div>
    <details class="paper-abstract">
      Smart contract comment generation has gained traction as a means to improve code comprehension and maintainability in blockchain systems. However, evaluating the quality of generated comments remains a challenge. Traditional metrics such as BLEU and ROUGE fail to capture domain-specific nuances, while human evaluation is costly and unscalable. In this paper, we present \texttt{evalSmarT}, a modular and extensible framework that leverages large language models (LLMs) as evaluators. The system supports over 400 evaluator configurations by combining approximately 40 LLMs with 10 prompting strategies. We demonstrate its application in benchmarking comment generation tools and selecting the most informative outputs. Our results show that prompt design significantly impacts alignment with human judgment, and that LLM-based evaluation offers a scalable and semantically rich alternative to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19299v3">The Impact of LoRA Adapters on LLMs for Clinical Text Classification Under Computational and Data Constraints</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted for publication in the IEEE Access
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to domain gap, limited data, and stringent hardware constraints. In this study, we evaluate four adapter techniques-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN) - equivalent to Low-Rank Adaptation (LoRA), for clinical note classification under real-world, resource-constrained conditions. All experiments were conducted on a single NVIDIA Quadro P620 GPU (2 GB VRAM, 512 CUDA cores, 1.386 TFLOPS FP32), limiting batch sizes to <8 sequences and maximum sequence length to 256 tokens. Our clinical corpus comprises only 580 000 tokens, several orders of magnitude smaller than standard LLM pre-training datasets. We fine-tuned three biomedical pre-trained LLMs (CamemBERT-bio, AliBERT, DrBERT) and two lightweight Transformer models trained from scratch. Results show that 1) adapter structures provide no consistent gains when fine-tuning biomedical LLMs under these constraints, and 2) simpler Transformers, with minimal parameter counts and training times under six hours, outperform adapter-augmented LLMs, which required over 1000 GPU-hours. Among adapters, GRN achieved the best metrics (accuracy, precision, recall, F1 = 0.88). These findings demonstrate that, in low-resource clinical settings with limited data and compute, lightweight Transformers trained from scratch offer a more practical and efficient solution than large LLMs, while GRN remains a viable adapter choice when minimal adaptation is needed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06003v3">LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Conference Version (ISCA'25). Fixed a typo
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference becomes resource-intensive, prompting a shift toward low-bit model weights to reduce the memory footprint and improve efficiency. Such low-bit LLMs necessitate the mixed-precision matrix multiplication (mpGEMM), an important yet underexplored operation involving the multiplication of lower-precision weights with higher-precision activations. Off-the-shelf hardware does not support this operation natively, leading to indirect, thus inefficient, dequantization-based implementations. In this paper, we study the lookup table (LUT)-based approach for mpGEMM and find that a conventional LUT implementation fails to achieve the promised gains. To unlock the full potential of LUT-based mpGEMM, we propose LUT Tensor Core, a software-hardware co-design for low-bit LLM inference. LUT Tensor Core differentiates itself from conventional LUT designs through: 1) software-based optimizations to minimize table precompute overhead and weight reinterpretation to reduce table storage; 2) a LUT-based Tensor Core hardware design with an elongated tiling shape to maximize table reuse and a bit-serial design to support diverse precision combinations in mpGEMM; 3) a new instruction set and compilation optimizations for LUT-based mpGEMM. LUT Tensor Core significantly outperforms existing pure software LUT implementations and achieves a 1.44$\times$ improvement in compute density and energy efficiency compared to previous state-of-the-art LUT-based accelerators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20674v1">LLM-Based Repair of Static Nullability Errors</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Modern Java projects increasingly adopt static analysis tools that prevent null-pointer exceptions by treating nullness as a type property. However, integrating such tools into large, existing codebases remains a significant challenge. While annotation inference can eliminate many errors automatically, a subset of residual errors -- typically a mix of real bugs and false positives -- often persist and can only be resolved via code changes. Manually addressing these errors is tedious and error-prone. Large language models (LLMs) offer a promising path toward automating these repairs, but naively-prompted LLMs often generate incorrect, contextually-inappropriate edits. Resolving a nullability error demands a deep understanding of how a symbol is used across the codebase, often spanning methods, classes, and packages. We present NullRepair, a system that integrates LLMs into a structured workflow for resolving the errors from a nullability checker. NullRepair's decision process follows a flowchart derived from manual analysis of 200 real-world errors. It leverages static analysis to identify safe and unsafe usage regions of symbols, using error-free usage examples to contextualize model prompts. Patches are generated through an iterative interaction with the LLM that incorporates project-wide context and decision logic. Our evaluation on 12 real-world Java projects shows that NullRepair resolves an average of 72% of the errors that remain after applying a state-of-the-art annotation inference technique. Unlike a naively-prompted LLM, NullRepair also largely preserves program semantics, with all unit tests passing in 10/12 projects after applying every edit proposed by NullRepair, and 98% or more tests passing in the remaining two projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20666v1">MIMII-Agent: Leveraging LLMs with Function Calling for Relative Evaluation of Anomalous Sound Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper proposes a method for generating machine-type-specific anomalies to evaluate the relative performance of unsupervised anomalous sound detection (UASD) systems across different machine types, even in the absence of real anomaly sound data. Conventional keyword-based data augmentation methods often produce unrealistic sounds due to their reliance on manually defined labels, limiting scalability as machine types and anomaly patterns diversify. Advanced audio generative models, such as MIMII-Gen, show promise but typically depend on anomalous training data, making them less effective when diverse anomalous examples are unavailable. To address these limitations, we propose a novel synthesis approach leveraging large language models (LLMs) to interpret textual descriptions of faults and automatically select audio transformation functions, converting normal machine sounds into diverse and plausible anomalous sounds. We validate this approach by evaluating a UASD system trained only on normal sounds from five machine types, using both real and synthetic anomaly data. Experimental results reveal consistent trends in relative detection difficulty across machine types between synthetic and real anomalies. This finding supports our hypothesis and highlights the effectiveness of the proposed LLM-based synthesis approach for relative evaluation of UASD systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20655v1">CoGrader: Transforming Instructors' Assessment of Project Reports through Collaborative LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Grading project reports are increasingly significant in today's educational landscape, where they serve as key assessments of students' comprehensive problem-solving abilities. However, it remains challenging due to the multifaceted evaluation criteria involved, such as creativity and peer-comparative achievement. Meanwhile, instructors often struggle to maintain fairness throughout the time-consuming grading process. Recent advances in AI, particularly large language models, have demonstrated potential for automating simpler grading tasks, such as assessing quizzes or basic writing quality. However, these tools often fall short when it comes to complex metrics, like design innovation and the practical application of knowledge, that require an instructor's educational insights into the class situation. To address this challenge, we conducted a formative study with six instructors and developed CoGrader, which introduces a novel grading workflow combining human-LLM collaborative metrics design, benchmarking, and AI-assisted feedback. CoGrader was found effective in improving grading efficiency and consistency while providing reliable peer-comparative feedback to students. We also discuss design insights and ethical considerations for the development of human-AI collaborative grading systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18181v2">SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by Design Automation Conference (DAC) 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13940v2">DrugPilot: LLM-based Parameterized Reasoning Agent for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 29 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) integrated with autonomous agents hold significant potential for advancing scientific discovery through automated reasoning and task execution. However, applying LLM agents to drug discovery is still constrained by challenges such as large-scale multimodal data processing, limited task automation, and poor support for domain-specific tools. To overcome these limitations, we introduce DrugPilot, a LLM-based agent system with a parameterized reasoning architecture designed for end-to-end scientific workflows in drug discovery. DrugPilot enables multi-stage research processes by integrating structured tool use with a novel parameterized memory pool. The memory pool converts heterogeneous data from both public sources and user-defined inputs into standardized representations. This design supports efficient multi-turn dialogue, reduces information loss during data exchange, and enhances complex scientific decision-making. To support training and benchmarking, we construct a drug instruction dataset covering eight core drug discovery tasks. Under the Berkeley function-calling benchmark, DrugPilot significantly outperforms state-of-the-art agents such as ReAct and LoT, achieving task completion rates of 98.0%, 93.5%, and 64.0% for simple, multi-tool, and multi-turn scenarios, respectively. These results highlight DrugPilot's potential as a versatile agent framework for computational science domains requiring automated, interactive, and data-integrated reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06102v3">SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      With the proliferation of Large Language Models (LLMs) in Business Intelligence (BI), existing solutions face critical challenges in industrial deployments: functionality deficiencies from legacy systems failing to meet evolving LLM-era user demands, interaction limitations from single-round SQL generation paradigms inadequate for multi-round clarification, and cost for domain adaptation arising from cross-domain methods migration. We present SiriusBI, a practical LLM-powered BI system addressing the challenges of industrial deployments through three key innovations: (a) An end-to-end architecture integrating multi-module coordination to overcome functionality gaps in legacy systems; (b) A multi-round dialogue with querying mechanism, consisting of semantic completion, knowledge-guided clarification, and proactive querying processes, to resolve interaction constraints in SQL generation; (c) A data-conditioned SQL generation method selection strategy that supports both an efficient one-step Fine-Tuning approach and a two-step method leveraging Semantic Intermediate Representation for low-cost cross-domain applications. Experiments on both real-world datasets and public benchmarks demonstrate the effectiveness of SiriusBI. User studies further confirm that SiriusBI enhances both productivity and user experience. As an independent service on Tencent's data platform, SiriusBI is deployed across finance, advertising, and cloud sectors, serving dozens of enterprise clients. It achieves over 93% accuracy in SQL generation and reduces data analysts' query time from minutes to seconds in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v1">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20527v1">SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      The demand for Large Language Models (LLMs) capable of sophisticated mathematical reasoning is growing across industries. However, the development of performant mathematical LLMs is critically bottlenecked by the scarcity of difficult, novel training data. We introduce \textbf{SAND-Math} (Synthetic Augmented Novel and Difficult Mathematics problems and solutions), a pipeline that addresses this by first generating high-quality problems from scratch and then systematically elevating their complexity via a new \textbf{Difficulty Hiking} step. We demonstrate the effectiveness of our approach through two key findings. First, augmenting a strong baseline with SAND-Math data significantly boosts performance, outperforming the next-best synthetic dataset by \textbf{$\uparrow$ 17.85 absolute points} on the AIME25 benchmark. Second, in a dedicated ablation study, we show our Difficulty Hiking process is highly effective: by increasing average problem difficulty from 5.02 to 5.98, this step lifts AIME25 performance from 46.38\% to 49.23\%. The full generation pipeline, final dataset, and a fine-tuned model form a practical and scalable toolkit for building more capable and efficient mathematical reasoning LLMs. SAND-Math dataset is released here: \href{https://huggingface.co/datasets/amd/SAND-MATH}{https://huggingface.co/datasets/amd/SAND-MATH}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20511v1">Beyond Class Tokens: LLM-guided Dominant Property Mining for Few-shot Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 11 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Few-shot Learning (FSL), which endeavors to develop the generalization ability for recognizing novel classes using only a few images, faces significant challenges due to data scarcity. Recent CLIP-like methods based on contrastive language-image pertaining mitigate the issue by leveraging textual representation of the class name for unseen image discovery. Despite the achieved success, simply aligning visual representations to class name embeddings would compromise the visual diversity for novel class discrimination. To this end, we proposed a novel Few-Shot Learning (FSL) method (BCT-CLIP) that explores \textbf{dominating properties} via contrastive learning beyond simply using class tokens. Through leveraging LLM-based prior knowledge, our method pushes forward FSL with comprehensive structural image representations, including both global category representation and the patch-aware property embeddings. In particular, we presented a novel multi-property generator (MPG) with patch-aware cross-attentions to generate multiple visual property tokens, a Large-Language Model (LLM)-assistant retrieval procedure with clustering-based pruning to obtain dominating property descriptions, and a new contrastive learning strategy for property-token learning. The superior performances on the 11 widely used datasets demonstrate that our investigation of dominating properties advances discriminative class-specific representation learning and few-shot classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20509v1">LLMs-guided adaptive compensator: Bringing Adaptivity to Automatic Control Systems with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      With rapid advances in code generation, reasoning, and problem-solving, Large Language Models (LLMs) are increasingly applied in robotics. Most existing work focuses on high-level tasks such as task decomposition. A few studies have explored the use of LLMs in feedback controller design; however, these efforts are restricted to overly simplified systems, fixed-structure gain tuning, and lack real-world validation. To further investigate LLMs in automatic control, this work targets a key subfield: adaptive control. Inspired by the framework of model reference adaptive control (MRAC), we propose an LLM-guided adaptive compensator framework that avoids designing controllers from scratch. Instead, the LLMs are prompted using the discrepancies between an unknown system and a reference system to design a compensator that aligns the response of the unknown system with that of the reference, thereby achieving adaptivity. Experiments evaluate five methods: LLM-guided adaptive compensator, LLM-guided adaptive controller, indirect adaptive control, learning-based adaptive control, and MRAC, on soft and humanoid robots in both simulated and real-world environments. Results show that the LLM-guided adaptive compensator outperforms traditional adaptive controllers and significantly reduces reasoning complexity compared to the LLM-guided adaptive controller. The Lyapunov-based analysis and reasoning-path inspection demonstrate that the LLM-guided adaptive compensator enables a more structured design process by transforming mathematical derivation into a reasoning task, while exhibiting strong generalizability, adaptability, and robustness. This study opens a new direction for applying LLMs in the field of automatic control, offering greater deployability and practicality compared to vision-language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06424v2">Training LLM-based Tutors to Improve Student Learning Outcomes in Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Published in AIED 2025: The 26th International Conference on Artificial Intelligence in Education
    </div>
    <details class="paper-abstract">
      Generative artificial intelligence (AI) has the potential to scale up personalized tutoring through large language models (LLMs). Recent AI tutors are adapted for the tutoring task by training or prompting LLMs to follow effective pedagogical principles, though they are not trained to maximize student learning throughout the course of a dialogue. Therefore, they may engage with students in a suboptimal way. We address this limitation by introducing an approach to train LLMs to generate tutor utterances that maximize the likelihood of student correctness, while still encouraging the model to follow good pedagogical practice. Specifically, we generate a set of candidate tutor utterances and score them using (1) an LLM-based student model to predict the chance of correct student responses and (2) a pedagogical rubric evaluated by GPT-4o. We then use the resulting data to train an open-source LLM, Llama 3.1 8B, using direct preference optimization. We show that tutor utterances generated by our model lead to significantly higher chances of correct student responses while maintaining the pedagogical quality of GPT-4o. We also conduct qualitative analyses and a human evaluation to demonstrate that our model generates high quality tutor utterances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21285v1">Curiosity by Design: An LLM-based Coding Assistant Asking Clarification Questions</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as coding assistants. However, the ambiguity of the developer's prompt often leads to incorrect code generation, as current models struggle to infer user intent without extensive prompt engineering or external context. This work aims to build an LLM-based coding assistant that mimics the human code review process by asking clarification questions when faced with ambiguous or under-specified queries. Our end-to-end system includes (1) a query classifier trained to detect unclear programming-related queries and (2) a fine-tuned LLM that generates clarification questions. Our evaluation shows that the fine-tuned LLM outperforms standard zero-shot prompting in generating useful clarification questions. Furthermore, our user study indicates that users find the clarification questions generated by our model to outperform the baseline, demonstrating that our coding assistant produces more accurate and helpful code responses compared to baseline coding assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21276v1">LeMix: Unified Scheduling for LLM Training and Inference on Multi-GPU Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by RTSS 2025
    </div>
    <details class="paper-abstract">
      Modern deployment of large language models (LLMs) frequently involves both inference serving and continuous retraining to stay aligned with evolving data and user feedback. Common practices separate these workloads onto distinct servers in isolated phases, causing substantial inefficiencies (e.g., GPU idleness) and delayed adaptation to new data in distributed settings. Our empirical analysis reveals that these inefficiencies stem from dynamic request arrivals during serving and workload heterogeneity in pipeline-parallel training. To address these challenges, we propose LeMix, a system for co-locating and managing concurrent LLM serving and training workloads. LeMix integrates offline profiling, execution prediction mechanisms, and runtime scheduling to dynamically adapt resource allocation based on workload characteristics and system conditions. By understanding task-specific behaviors and co-execution interference across shared nodes, LeMix improves utilization and serving quality without compromising serving responsiveness. Our evaluation shows that LeMix improves throughput by up to 3.53x, reduces inference loss by up to 0.61x, and delivers up to 2.12x higher response time SLO attainment over traditional separate setups. To our knowledge, this is the first work to uncover and exploit the opportunities of joint LLM inference and training, paving the way for more resource-efficient deployment of LLMs in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21257v1">CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Research Track, 24th International Semantic Web Conference (ISWC 2025), November 2-6, 2025, Nara, Japan
    </div>
    <details class="paper-abstract">
      Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21199v1">Advancing Compositional LLM Reasoning with Structured Task Relations in Interactive Multimodal Communications</a></div>
    <div class="paper-meta">
      📅 2025-07-28
      | 💬 Accepted by IEEE JSAC. This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Interactive multimodal applications (IMAs), such as route planning in the Internet of Vehicles, enrich users' personalized experiences by integrating various forms of data over wireless networks. Recent advances in large language models (LLMs) utilize mixture-of-experts (MoE) mechanisms to empower multiple IMAs, with each LLM trained individually for a specific task that presents different business workflows. In contrast to existing approaches that rely on multiple LLMs for IMAs, this paper presents a novel paradigm that accomplishes various IMAs using a single compositional LLM over wireless networks. The two primary challenges include 1) guiding a single LLM to adapt to diverse IMA objectives and 2) ensuring the flexibility and efficiency of the LLM in resource-constrained mobile environments. To tackle the first challenge, we propose ContextLoRA, a novel method that guides an LLM to learn the rich structured context among IMAs by constructing a task dependency graph. We partition the learnable parameter matrix of neural layers for each IMA to facilitate LLM composition. Then, we develop a step-by-step fine-tuning procedure guided by task relations, including training, freezing, and masking phases. This allows the LLM to learn to reason among tasks for better adaptation, capturing the latent dependencies between tasks. For the second challenge, we introduce ContextGear, a scheduling strategy to optimize the training procedure of ContextLoRA, aiming to minimize computational and communication costs through a strategic grouping mechanism. Experiments on three benchmarks show the superiority of the proposed ContextLoRA and ContextGear. Furthermore, we prototype our proposed paradigm on a real-world wireless testbed, demonstrating its practical applicability for various IMAs. We will release our code to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22086v1">TypyBench: Evaluating LLM Type Inference for Untyped Python Repositories</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Type inference for dynamic languages like Python is a persistent challenge in software engineering. While large language models (LLMs) have shown promise in code understanding, their type inference capabilities remain underexplored. We introduce TypyBench, a benchmark designed to evaluate LLMs' type inference across entire Python repositories. TypyBench features two novel metrics: TypeSim, which captures nuanced semantic relationships between predicted and ground truth types, and TypeCheck, which assesses type consistency across codebases. Our evaluation of various LLMs on a curated dataset of 50 high-quality Python repositories reveals that, although LLMs achieve decent TypeSim scores, they struggle with complex nested types and exhibit significant type consistency errors. These findings suggest that future research should shift focus from improving type similarity to addressing repository-level consistency. TypyBench provides a foundation for this new direction, offering insights into model performance across different type complexities and usage contexts. Our code and data are available at https://github.com/typybench/typybench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22171v1">Enhancing Jailbreak Attacks on LLMs via Persona Prompts</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2507.19353v1">Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-28
    </div>
    <details class="paper-abstract">
      Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02622v2">Exploring undercurrents of learning tensions in an LLM-enhanced landscape: A student-centered qualitative perspective on LLM vs Search</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are transforming how students learn by providing readily available tools that can quickly augment or complete various learning activities with non-trivial performance. Similar paradigm shifts have occurred in the past with the introduction of search engines and Wikipedia, which replaced or supplemented traditional information sources such as libraries and books. This study investigates the potential for LLMs to represent the next shift in learning, focusing on their role in information discovery and synthesis compared to existing technologies, such as search engines. Using a within-subjects, counterbalanced design, participants learned new topics using a search engine (Google) and an LLM (ChatGPT). Post-task follow-up interviews explored students' reflections, preferences, pain points, and overall perceptions. We present analysis of their responses that show nuanced insights into when, why, and how students prefer LLMs over search engines, offering implications for educators, policymakers, and technology developers navigating the evolving educational landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14133v3">Self-Regularization with Sparse Autoencoders for Controllable LLM-based Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Accepted by SIGKDD 2025
    </div>
    <details class="paper-abstract">
      Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impact of these unintended features on classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed self-regularization framework can improve the classifier's generalizability by regularizing those features that are not semantically correlated to the task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. The code and data are publicly available at https://github.com/JacksonWuxs/Controllable_LLM_Classifier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00946v3">A Review of LLM-Assisted Ideation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      We present a comprehensive, in-depth review of ideation assisted by large language models (LLMs), highlighting emerging trends and identifying unaddressed research gaps. In total, we examined 61 studies investigating the application of LLMs in both group and individual ideation processes. From these studies, we derived the Hourglass Ideation Framework for LLM-assisted ideation, comprising three phases and seven key ideation stages, which served as the basis for our systematic survey. Our analysis reveals that LLMs are most frequently used for idea generation and refinement, but their use in scope specification, foundational material structuring and multi-idea evaluation and selection remains limited. We provide our findings in extensive tabular and online formats. These catalogues detail research on LLM-assisted, purely LLM-based, and human-only activities across the seven ideation stages for each of the 61 studies. These also detail creative domains, publication outlets, interaction designs, user study designs, and assessment methods. Our analysis of system interaction design reveals a predominant focus on supporting individual ideation activities and text-based interaction, with a growing trend of incorporating multimedia elements. However, in group ideation, tools and interaction modalities targeting both synchronous and asynchronous collaboration are much scarcer. We synthesize the primary findings of our review and outline promising directions for future research in LLM-assisted ideation. We hope this review will help researchers quickly gain an overview of this rapidly expanding area, efficiently locate relevant work, and identify underexplored areas for further investigation. In addition, we believe the framework we present here will form the basis for the development of future problem and solution space taxonomies, and methodologies for LLM-assisted ideation development and use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20352v1">RMTBench: Benchmarking LLMs Through Multi-Turn User-Centric Role-Playing</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have shown outstanding potential for role-playing applications. Evaluating these capabilities is becoming crucial yet remains challenging. Existing benchmarks mostly adopt a \textbf{character-centric} approach, simplify user-character interactions to isolated Q&A tasks, and fail to reflect real-world applications. To address this limitation, we introduce RMTBench, a comprehensive \textbf{user-centric} bilingual role-playing benchmark featuring 80 diverse characters and over 8,000 dialogue rounds. RMTBench includes custom characters with detailed backgrounds and abstract characters defined by simple traits, enabling evaluation across various user scenarios. Our benchmark constructs dialogues based on explicit user motivations rather than character descriptions, ensuring alignment with practical user applications. Furthermore, we construct an authentic multi-turn dialogue simulation mechanism. With carefully selected evaluation dimensions and LLM-based scoring, this mechanism captures the complex intention of conversations between the user and the character. By shifting focus from character background to user intention fulfillment, RMTBench bridges the gap between academic evaluation and practical deployment requirements, offering a more effective framework for assessing role-playing capabilities in LLMs. All code and datasets will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04626v2">Heterogeneous User Modeling for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Accepted by RecSys 2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for recommendation has demonstrated notable success in various domains, showcasing their potential for open-domain recommendation. A key challenge to advancing open-domain recommendation lies in effectively modeling user preferences from users' heterogeneous behaviors across multiple domains. Existing approaches, including ID-based and semantic-based modeling, struggle with poor generalization, an inability to compress noisy interactions effectively, and the domain seesaw phenomenon. To address these challenges, we propose a Heterogeneous User Modeling (HUM) method, which incorporates a compression enhancer and a robustness enhancer for LLM-based recommendation. The compression enhancer uses a customized prompt to compress heterogeneous behaviors into a tailored token, while a masking mechanism enhances cross-domain knowledge extraction and understanding. The robustness enhancer introduces a domain importance score to mitigate the domain seesaw phenomenon by guiding domain optimization. Extensive experiments on heterogeneous datasets validate that HUM effectively models user heterogeneity by achieving both high efficacy and robustness, leading to superior performance in open-domain recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20300v1">Talking-to-Build: How LLM-Assisted Interface Shapes Player Performance and Experience in Minecraft</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      With large language models (LLMs) on the rise, in-game interactions are shifting from rigid commands to natural conversations. However, the impacts of LLMs on player performance and game experience remain underexplored. This work explores LLM's role as a co-builder during gameplay, examining its impact on task performance, usability, and player experience. Using Minecraft as a sandbox, we present an LLM-assisted interface that engages players through natural language, aiming to facilitate creativity and simplify complex gaming commands. We conducted a mixed-methods study with 30 participants, comparing LLM-assisted and command-based interfaces across simple and complex game tasks. Quantitative and qualitative analyses reveal that the LLM-assisted interface significantly improves player performance, engagement, and overall game experience. Additionally, task complexity has a notable effect on player performance and experience across both interfaces. Our findings highlight the potential of LLM-assisted interfaces to revolutionize virtual experiences, emphasizing the importance of balancing intuitiveness with predictability, transparency, and user agency in AI-driven, multimodal gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11150v3">ADAgent: LLM Agent for Alzheimer's Disease Analysis with Collaborative Coordinator</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Alzheimer's disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20278v1">MoL-RL: Distilling Multi-Step Environmental Feedback into LLMs for Feedback-Independent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 12pages,3figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant challenges in effectively leveraging sequential environmental feedback (EF) signals, such as natural language evaluations, for feedback-independent chain-of-thought (CoT) reasoning. Existing approaches either convert EF into scalar rewards, losing rich contextual information, or employ refinement datasets, failing to exploit the multi-step and discrete nature of EF interactions. To address these limitations, we propose MoL-RL, a novel training paradigm that integrates multi-step EF signals into LLMs through a dual-objective optimization framework. Our method combines MoL (Mixture-of-Losses) continual training, which decouples domain-specific EF signals (optimized via cross-entropy loss) and general language capabilities (preserved via Kullback-Leibler divergence), with GRPO-based post-training to distill sequential EF interactions into single-step inferences. This synergy enables robust feedback-independent reasoning without relying on external feedback loops. Experimental results on mathematical reasoning (MATH-500, AIME24/AIME25) and code generation (CodeAgent-Test) benchmarks demonstrate that MoL-RL achieves state-of-the-art performance with the Qwen3-8B model, while maintaining strong generalization across model scales (Qwen3-4B). This work provides a promising approach for leveraging multi-step textual feedback to enhance LLMs' reasoning capabilities in diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13632v4">DoubleDipper: Improving Long-Context LLMs via Context Recycling</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Despite recent advancements in Large Language Models (LLMs), their performance on tasks involving long contexts remains sub-optimal. In this work, we propose DoubleDipper, a novel In-Context-Learning method that automatically generates few-shot examples for long context QA tasks by recycling contexts. Specifically, given a long input context (1-3k tokens) and a query, we generate additional query-output pairs from the given context as few-shot examples, while introducing the context only once. This ensures that the demonstrations are leveraging the same context as the target query while only adding a small number of tokens to the prompt. We further enhance each demonstration by instructing the model to explicitly identify the relevant paragraphs before the answer, which improves performance while providing fine-grained attribution to the answer source. We apply our method on multiple LLMs and obtain substantial improvements (+16 absolute points on average across models) on various QA datasets with long context. Surprisingly, despite introducing only single-hop ICL examples, LLMs successfully generalize to multi-hop long-context QA using our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07840v3">Understanding Learner-LLM Chatbot Interactions and the Impact of Prompting Guidelines</a></div>
    <div class="paper-meta">
      📅 2025-07-27
      | 💬 Long paper accepted for AIED 2025, the 26th International Conference on Artificial Intelligence in Education, July 22 - 26, 2025, Palermo, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed human-computer interaction by enabling natural language-based communication with AI-powered chatbots. These models are designed to be intuitive and user-friendly, allowing users to articulate requests with minimal effort. However, despite their accessibility, studies reveal that users often struggle with effective prompting, resulting in inefficient responses. Existing research has highlighted both the limitations of LLMs in interpreting vague or poorly structured prompts and the difficulties users face in crafting precise queries. This study investigates learner-AI interactions through an educational experiment in which participants receive structured guidance on effective prompting. We introduce and compare three types of prompting guidelines: a task-specific framework developed through a structured methodology and two baseline approaches. To assess user behavior and prompting efficacy, we analyze a dataset of 642 interactions from 107 users. Using Von NeuMidas, an extended pragmatic annotation schema for LLM interaction analysis, we categorize common prompting errors and identify recurring behavioral patterns. We then evaluate the impact of different guidelines by examining changes in user behavior, adherence to prompting strategies, and the overall quality of AI-generated responses. Our findings provide a deeper understanding of how users engage with LLMs and the role of structured prompting guidance in enhancing AI-assisted communication. By comparing different instructional frameworks, we offer insights into more effective approaches for improving user competency in AI interactions, with implications for AI literacy, chatbot usability, and the design of more responsive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20208v1">IQ Test for LLMs: An Evaluation Framework for Uncovering Core Skills in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Current evaluations of large language models (LLMs) rely on benchmark scores, but it is difficult to interpret what these individual scores reveal about a model's overall skills. Specifically, as a community we lack understanding of how tasks relate to one another, what they measure in common, how they differ, or which ones are redundant. As a result, models are often assessed via a single score averaged across benchmarks, an approach that fails to capture the models' wholistic strengths and limitations. Here, we propose a new evaluation paradigm that uses factor analysis to identify latent skills driving performance across benchmarks. We apply this method to a comprehensive new leaderboard showcasing the performance of 60 LLMs on 44 tasks, and identify a small set of latent skills that largely explain performance. Finally, we turn these insights into practical tools that identify redundant tasks, aid in model selection, and profile models along each latent skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06774v2">Checklist Engineering Empowers Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Automated text evaluation has long been a central issue in Natural Language Processing (NLP). Recently, the field has shifted toward using Large Language Models (LLMs) as evaluators-a trend known as the LLM-as-a-Judge paradigm. While promising and easily adaptable across tasks, this approach has seen limited exploration in multilingual contexts. Existing multilingual studies often rely on proprietary models or require extensive training data for fine-tuning, raising concerns about cost, time, and efficiency. In this paper, we propose Checklist Engineering based LLM-as-a-Judge (CE-Judge), a training-free framework that uses checklist intuition for multilingual evaluation with an open-source model. Experiments across multiple languages and three benchmark datasets, under both pointwise and pairwise settings, show that our method generally surpasses the baselines and performs on par with the GPT-4o model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20152v1">Goal Alignment in LLM-Based User Simulators for Conversational AI</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      User simulators are essential to conversational AI, enabling scalable agent development and evaluation through simulated interactions. While current Large Language Models (LLMs) have advanced user simulation capabilities, we reveal that they struggle to consistently demonstrate goal-oriented behavior across multi-turn conversations--a critical limitation that compromises their reliability in downstream applications. We introduce User Goal State Tracking (UGST), a novel framework that tracks user goal progression throughout conversations. Leveraging UGST, we present a three-stage methodology for developing user simulators that can autonomously track goal progression and reason to generate goal-aligned responses. Moreover, we establish comprehensive evaluation metrics for measuring goal alignment in user simulators, and demonstrate that our approach yields substantial improvements across two benchmarks (MultiWOZ 2.4 and {\tau}-Bench). Our contributions address a critical gap in conversational AI and establish UGST as an essential framework for developing goal-aligned user simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20147v1">Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Session-based recommendation (SBR) is mainly based on anonymous user interaction sequences to recommend the items that the next user is most likely to click. Currently, the most popular and high-performing SBR methods primarily leverage graph neural networks (GNNs), which model session sequences as graph-structured data to effectively capture user intent. However, most GNNs-based SBR methods primarily focus on modeling the ID sequence information of session sequences, while neglecting the rich semantic information embedded within them. This limitation significantly hampers model's ability to accurately infer users' true intention. To address above challenge, this paper proposes a novel SBR approach called Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation (LLM-DMsRec). The method utilizes a pre-trained GNN model to select the top-k items as candidate item sets and designs prompts along with a large language model (LLM) to infer multi-semantic intents from these candidate items. Specifically, we propose an alignment mechanism that effectively integrates the semantic intent inferred by the LLM with the structural intent captured by GNNs. Extensive experiments conducted on the Beauty and ML-1M datasets demonstrate that the proposed method can be seamlessly integrated into GNNs framework, significantly enhancing its recommendation performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14066v3">TurboSpec: Closed-loop Speculation Control System for Optimizing LLM Serving Goodput</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) serving systems batch concurrent user requests to achieve efficient serving. However, in real-world deployments, such inter-request parallelism from batching is often limited by external factors such as low request rates or memory constraints. Recent works focus on intra-request parallelism from speculative decoding as a solution to this problem. Unfortunately, benefits from intra-request parallelism are often fragile, as speculative decoding causes overhead, and speculated tokens may miss. We observe that speculative decoding may degrade LLM serving performance if added naively without tuning to the incoming requests and the speculation method. To alleviate the need for expert tuning and make speculative decoding more robust, we present TurboSpec, a speculation control system that automatically profiles the execution environment and utilizes a feedback-based algorithm to dynamically adjust the amount of intra-request parallelism in LLM serving. TurboSpec predicts "goodput" - the amount of successfully generated tokens - to evaluate and adjust intra-request parallelism amount to that with the highest goodput in runtime. We implement TurboSpec on a real-world LLM serving system vLLM and demonstrate its effectiveness across diverse workloads and hardware configurations, providing consistent performance improvements across all test scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21193v1">Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      Next generation Radio Access Networks (RANs) introduce programmability, intelligence, and near real-time control through intelligent controllers, enabling enhanced security within the RAN and across broader 5G/6G infrastructures. This paper presents a comprehensive survey highlighting opportunities, challenges, and research gaps for Large Language Models (LLMs)-assisted explainable (XAI) intrusion detection (IDS) for secure future RAN environments. Motivated by this, we propose an LLM interpretable anomaly-based detection system for distributed denial-of-service (DDoS) attacks using multivariate time series key performance measures (KPMs), extracted from E2 nodes, within the Near Real-Time RAN Intelligent Controller (Near-RT RIC). An LSTM-based model is trained to identify malicious User Equipment (UE) behavior based on these KPMs. To enhance transparency, we apply post-hoc local explainability methods such as LIME and SHAP to interpret individual predictions. Furthermore, LLMs are employed to convert technical explanations into natural-language insights accessible to non-expert users. Experimental results on real 5G network KPMs demonstrate that our framework achieves high detection accuracy (F1-score > 0.96) while delivering actionable and interpretable outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21188v1">Embeddings to Diagnosis: Latent Fragility under Agentic Perturbations in Clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-27
    </div>
    <details class="paper-abstract">
      LLMs for clinical decision support often fail under small but clinically meaningful input shifts such as masking a symptom or negating a finding, despite high performance on static benchmarks. These reasoning failures frequently go undetected by standard NLP metrics, which are insensitive to latent representation shifts that drive diagnosis instability. We propose a geometry-aware evaluation framework, LAPD (Latent Agentic Perturbation Diagnostics), which systematically probes the latent robustness of clinical LLMs under structured adversarial edits. Within this framework, we introduce Latent Diagnosis Flip Rate (LDFR), a model-agnostic diagnostic signal that captures representational instability when embeddings cross decision boundaries in PCA-reduced latent space. Clinical notes are generated using a structured prompting pipeline grounded in diagnostic reasoning, then perturbed along four axes: masking, negation, synonym replacement, and numeric variation to simulate common ambiguities and omissions. We compute LDFR across both foundation and clinical LLMs, finding that latent fragility emerges even under minimal surface-level changes. Finally, we validate our findings on 90 real clinical notes from the DiReCT benchmark (MIMIC-IV), confirming the generalizability of LDFR beyond synthetic settings. Our results reveal a persistent gap between surface robustness and semantic stability, underscoring the importance of geometry-aware auditing in safety-critical clinical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12088v2">Risks & Benefits of LLMs & GenAI for Platform Integrity, Healthcare Diagnostics, Financial Trust and Compliance, Cybersecurity, Privacy & AI Safety: A Comprehensive Survey, Roadmap & Implementation Blueprint</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, and Copilot (by OpenAI, Anthropic, Google, Meta, and Microsoft, respectively), are reshaping digital platforms and app ecosystems while introducing critical challenges in cybersecurity, privacy, and platform integrity. Our analysis reveals alarming trends: LLM-assisted malware is projected to rise from 2% (2021) to 50% (2025); AI-generated Google reviews grew nearly tenfold (1.2% in 2021 to 12.21% in 2023, expected to reach 30% by 2025); AI scam reports surged 456%; misinformation sites increased over 1500%; and deepfake attacks are projected to rise over 900% in 2025. In finance, LLM-driven threats like synthetic identity fraud and AI-generated scams are accelerating. Platforms such as JPMorgan Chase, Stripe, and Plaid deploy LLMs for fraud detection, regulation parsing, and KYC/AML automation, reducing fraud loss by up to 21% and accelerating onboarding by 40-60%. LLM-facilitated code development has driven mobile app submissions from 1.8 million (2020) to 3.0 million (2024), projected to reach 3.6 million (2025). To address AI threats, platforms like Google Play, Apple App Store, GitHub Copilot, TikTok, Facebook, and Amazon deploy LLM-based defenses, highlighting their dual nature as both threat sources and mitigation tools. In clinical diagnostics, LLMs raise concerns about accuracy, bias, and safety, necessitating strong governance. Drawing on 445 references, this paper surveys LLM/GenAI and proposes a strategic roadmap and operational blueprint integrating policy auditing (such as CCPA and GDPR compliance), fraud detection, and demonstrates an advanced LLM-DA stack with modular components, multi-LLM routing, agentic memory, and governance layers. We provide actionable insights, best practices, and real-world case studies for scalable trust and responsible innovation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20067v1">PITA: Preference-Guided Inference-Time Alignment for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Inference-time alignment enables large language models (LLMs) to generate outputs aligned with end-user preferences without further training. Recent post-training methods achieve this by using small guidance models to modify token generation during inference. These methods typically optimize a reward function KL-regularized by the original LLM taken as the reference policy. A critical limitation, however, is their dependence on a pre-trained reward model, which requires fitting to human preference feedback--a potentially unstable process. In contrast, we introduce PITA, a novel framework that integrates preference feedback directly into the LLM's token generation, eliminating the need for a reward model. PITA learns a small preference-based guidance policy to modify token probabilities at inference time without LLM fine-tuning, reducing computational cost and bypassing the pre-trained reward model dependency. The problem is framed as identifying an underlying preference distribution, solved through stochastic search and iterative refinement of the preference-based guidance model. We evaluate PITA across diverse tasks, including mathematical reasoning and sentiment classification, demonstrating its effectiveness in aligning LLM outputs with user preferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20059v1">RAG in the Wild: On the (In)effectiveness of LLMs with Mixture-of-Knowledge Retrieval Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Work in Progress. Code will be published at: https://github.com/ritaranx/RAG_in_the_Wild
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge retrieved at inference time. While RAG demonstrates strong performance on benchmarks largely derived from general-domain corpora like Wikipedia, its effectiveness under realistic, diverse retrieval scenarios remains underexplored. We evaluated RAG systems using MassiveDS, a large-scale datastore with mixture of knowledge, and identified critical limitations: retrieval mainly benefits smaller models, rerankers add minimal value, and no single retrieval source consistently excels. Moreover, current LLMs struggle to route queries across heterogeneous knowledge sources. These findings highlight the need for adaptive retrieval strategies before deploying RAG in real-world settings. Our code and data can be found at https://github.com/ritaranx/RAG_in_the_Wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02456v2">Colombian Waitresses y Jueces canadienses: Gender and Country Biases in Occupation Recommendations from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Workshop on Gender Bias in Natural Language Processing at ACL 2025
    </div>
    <details class="paper-abstract">
      One of the goals of fairness research in NLP is to measure and mitigate stereotypical biases that are propagated by NLP systems. However, such work tends to focus on single axes of bias (most often gender) and the English language. Addressing these limitations, we contribute the first study of multilingual intersecting country and gender biases, with a focus on occupation recommendations generated by large language models. We construct a benchmark of prompts in English, Spanish and German, where we systematically vary country and gender, using 25 countries and four pronoun sets. Then, we evaluate a suite of 5 Llama-based models on this benchmark, finding that LLMs encode significant gender and country biases. Notably, we find that even when models show parity for gender or country individually, intersectional occupational biases based on both country and gender persist. We also show that the prompting language significantly affects bias, and instruction-tuned models consistently demonstrate the lowest and most stable levels of bias. Our findings highlight the need for fairness researchers to use intersectional and multilingual lenses in their work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19980v1">Exploring LLM Autoscoring Reliability in Large-Scale Writing Assessments Using Generalizability Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      This study investigates the estimation of reliability for large language models (LLMs) in scoring writing tasks from the AP Chinese Language and Culture Exam. Using generalizability theory, the research evaluates and compares score consistency between human and AI raters across two types of AP Chinese free-response writing tasks: story narration and email response. These essays were independently scored by two trained human raters and seven AI raters. Each essay received four scores: one holistic score and three analytic scores corresponding to the domains of task completion, delivery, and language use. Results indicate that although human raters produced more reliable scores overall, LLMs demonstrated reasonable consistency under certain conditions, particularly for story narration tasks. Composite scoring that incorporates both human and AI raters improved reliability, which supports that hybrid scoring models may offer benefits for large-scale writing assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09720v3">NestQuant: Nested Lattice Quantization for Matrix Products and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 23 pages; Accepted at the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent works have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Metas SpinQuant (perplexity 7.3), OstQuant (7.3) and QuaRot (8.2). Comparisons on bigger models (up to 70B) and on various LLM evaluation benchmarks confirm uniform superiority of NestQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19939v1">LLMControl: Grounded Control of Text-to-Image Diffusion-based Synthesis with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Recent spatial control methods for text-to-image (T2I) diffusion models have shown compelling results. However, these methods still fail to precisely follow the control conditions and generate the corresponding images, especially when encountering the textual prompts that contain multiple objects or have complex spatial compositions. In this work, we present a LLM-guided framework called LLM\_Control to address the challenges of the controllable T2I generation task. By improving grounding capabilities, LLM\_Control is introduced to accurately modulate the pre-trained diffusion models, where visual conditions and textual prompts influence the structures and appearance generation in a complementary way. We utilize the multimodal LLM as a global controller to arrange spatial layouts, augment semantic descriptions and bind object attributes. The obtained control signals are injected into the denoising network to refocus and enhance attention maps according to novel sampling constraints. Extensive qualitative and quantitative experiments have demonstrated that LLM\_Control achieves competitive synthesis quality compared to other state-of-the-art methods across various pre-trained T2I models. It is noteworthy that LLM\_Control allows the challenging input conditions on which most of the existing methods
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19899v1">A Gold Standard Dataset and Evaluation Framework for Depression Detection and Explanation in Social Media using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Early detection of depression from online social media posts holds promise for providing timely mental health interventions. In this work, we present a high-quality, expert-annotated dataset of 1,017 social media posts labeled with depressive spans and mapped to 12 depression symptom categories. Unlike prior datasets that primarily offer coarse post-level labels \cite{cohan-etal-2018-smhd}, our dataset enables fine-grained evaluation of both model predictions and generated explanations. We develop an evaluation framework that leverages this clinically grounded dataset to assess the faithfulness and quality of natural language explanations generated by large language models (LLMs). Through carefully designed prompting strategies, including zero-shot and few-shot approaches with domain-adapted examples, we evaluate state-of-the-art proprietary LLMs including GPT-4.1, Gemini 2.5 Pro, and Claude 3.7 Sonnet. Our comprehensive empirical analysis reveals significant differences in how these models perform on clinical explanation tasks, with zero-shot and few-shot prompting. Our findings underscore the value of human expertise in guiding LLM behavior and offer a step toward safer, more transparent AI systems for psychological well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19855v1">Inducing Causal World Models in LLMs for Zero-Shot Physical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 12 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their advanced linguistic capabilities, fundamentally lack an intuitive understanding of physical dynamics, which limits their effectiveness in real-world scenarios that require causal reasoning. In this paper, we introduce Causal World Model Induction (CWMI), a novel framework designed to embed an explicit model of causal physics within an LLM. Our approach incorporates a dedicated Causal Physics Module (CPM) and a new training objective called Causal Intervention Loss, encouraging the model to learn cause-and-effect relationships from multimodal data. By training the model to predict the outcomes of hypothetical interventions instead of merely capturing statistical correlations, CWMI develops a robust internal representation of physical laws. Experimental results show that CWMI significantly outperforms state-of-the-art LLMs on zero-shot physical reasoning tasks, including the PIQA benchmark and our newly proposed PhysiCa-Bench dataset. These findings demonstrate that inducing a causal world model is a critical step toward more reliable and generalizable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19845v1">MegatronApp: Efficient and Comprehensive Management on Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      The rapid escalation in the parameter count of large language models (LLMs) has transformed model training from a single-node endeavor into a highly intricate, cross-node activity. While frameworks such as Megatron-LM successfully integrate tensor (TP), pipeline (PP), and data (DP) parallelism to enable trillion-parameter training, they simultaneously expose practitioners to unprecedented systems-level challenges in performance optimization, diagnosis, and interpretability. MegatronApp is an open-source toolchain expressly designed to meet these challenges. It introduces four orthogonal, yet seamlessly composable modules--MegaScan, MegaFBD, MegaDPP, and MegaScope--that collectively elevate the reliability, efficiency, and transparency of production-scale training. This paper presents the motivation, architecture, and distinctive contributions of each module, and elucidates how their synergistic integration augments the Megatron-LM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19823v1">HCAttention: Extreme KV Cache Compression via Heterogeneous Attention Computing for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Processing long-context inputs with large language models presents a significant challenge due to the enormous memory requirements of the Key-Value (KV) cache during inference. Existing KV cache compression methods exhibit noticeable performance degradation when memory is reduced by more than 85%. Additionally, strategies that leverage GPU-CPU collaboration for approximate attention remain underexplored in this setting. We propose HCAttention, a heterogeneous attention computation framework that integrates key quantization, value offloading, and dynamic KV eviction to enable efficient inference under extreme memory constraints. The method is compatible with existing transformer architectures and does not require model fine-tuning. Experimental results on the LongBench benchmark demonstrate that our approach preserves the accuracy of full-attention model while shrinking the KV cache memory footprint to 25% of its original size. Remarkably, it stays competitive with only 12.5% of the cache, setting a new state-of-the-art in LLM KV cache compression. To the best of our knowledge, HCAttention is the first to extend the Llama-3-8B model to process 4 million tokens on a single A100 GPU with 80GB memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19749v1">Can LLMs Solve ASP Problems? Insights from a Benchmarking Study (Extended Version)</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 Accepted for publication at the 22nd International Conference on Principles of Knowledge Representation and Reasoning (KR 2025). The code is available at https://github.com/HomuraT/ASPBench
    </div>
    <details class="paper-abstract">
      Answer Set Programming (ASP) is a powerful paradigm for non-monotonic reasoning. Recently, large language models (LLMs) have demonstrated promising capabilities in logical reasoning. Despite this potential, current evaluations of LLM capabilities in ASP are often limited. Existing works normally employ overly simplified ASP programs, do not support negation, disjunction, or multiple answer sets. Furthermore, there is a lack of benchmarks that introduce tasks specifically designed for ASP solving. To bridge this gap, we introduce ASPBench, a comprehensive ASP benchmark, including three ASP specific tasks: ASP entailment, answer set verification, and answer set computation. Our extensive evaluations on ASPBench reveal that while 14 state-of-the-art LLMs, including \emph{deepseek-r1}, \emph{o4-mini}, and \emph{gemini-2.5-flash-thinking}, perform relatively well on the first two simpler tasks, they struggle with answer set computation, which is the core of ASP solving. These findings offer insights into the current limitations of LLMs in ASP solving. This highlights the need for new approaches that integrate symbolic reasoning capabilities more effectively. The code and dataset are available at https://github.com/HomuraT/ASPBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19747v1">TokenBlowUp: Resolving Representational Singularities in LLM Token Spaces via Monoidal Transformations</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Recent work has provided compelling evidence challenging the foundational manifold hypothesis for the token embedding spaces of Large Language Models (LLMs). These findings reveal the presence of geometric singularities around polysemous tokens, which can lead to representational instability. Existing methodologies, which presuppose a smooth data manifold, are ill-equipped to address such intrinsic structural flaws. In this paper, we formalize this problem in the language of scheme theory and propose a rigorous resolution by applying the scheme-theoretic blow-up at each singular point. This procedure replaces a singular point in the ambient affine scheme with its exceptional divisor, which we identify as a canonical geometric space -- a projective space of directions -- that houses the disambiguated semantic meanings of the token. This process of ``representational desingularization'' constructs a new geometric landscape for embeddings. We prove a formal theorem guaranteeing the geometric regularization of this new space, showing that the original pathologies are resolved. Finally, we outline the architectural implications of our framework, arguing for a paradigm shift from static look-ups to dynamic, geometrically-grounded computation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19737v1">Predicting Human Mobility in Disasters via LLM-Enhanced Cross-City Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      The vulnerability of cities to natural disasters has increased with urbanization and climate change, making it more important to predict human mobility in the disaster scenarios for downstream tasks including location-based early disaster warning and pre-allocating rescue resources, etc. However, existing human mobility prediction models are mainly designed for normal scenarios, and fail to adapt to disaster scenarios due to the shift of human mobility patterns under disaster. To address this issue, we introduce \textbf{DisasterMobLLM}, a mobility prediction framework for disaster scenarios that can be integrated into existing deep mobility prediction methods by leveraging LLMs to model the mobility intention and transferring the common knowledge of how different disasters affect mobility intentions between cities. This framework utilizes a RAG-Enhanced Intention Predictor to forecast the next intention, refines it with an LLM-based Intention Refiner, and then maps the intention to an exact location using an Intention-Modulated Location Predictor. Extensive experiments illustrate that DisasterMobLLM can achieve a 32.8\% improvement in terms of Acc@1 and a 35.0\% improvement in terms of the F1-score of predicting immobility compared to the baselines. The code is available at https://github.com/tsinghua-fib-lab/DisasterMobLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21179v1">LLM-Adapted Interpretation Framework for Machine Learning Models</a></div>
    <div class="paper-meta">
      📅 2025-07-26
      | 💬 11 pages, 8 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Background & Aims: High-performance machine learning models like XGBoost are often "black boxes," limiting their clinical adoption due to a lack of interpretability. This study aims to bridge the gap between predictive accuracy and narrative transparency for sarcopenia risk assessment. Methods: We propose the LLM-Adapted Interpretation Framework (LAI-ML), a novel knowledge distillation architecture. LAI-ML transforms feature attributions from a trained XGBoost model into a probabilistic format using specialized techniques (HAGA and CACS). A Large Language Model (LLM), guided by a reinforcement learning loop and case-based retrieval, then generates data-faithful diagnostic narratives. Results: The LAI-ML framework achieved 83% prediction accuracy, significantly outperforming the baseline XGBoost model, 13% higher. Notably, the LLM not only replicated the teacher model's logic but also corrected its predictions in 21.7% of discordant cases, demonstrating enhanced reasoning. Conclusion: LAI-ML effectively translates opaque model predictions into trustworthy and interpretable clinical insights, offering a deployable solution to the "black-box" problem in medical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21176v1">Tell Me You're Biased Without Telling Me You're Biased -- Toward Revealing Implicit Biases in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) that are used in medical applications are known to show biased and unfair patterns. Prior to adopting these in clinical decision-making applications, it is crucial to identify these bias patterns to enable effective mitigation of their impact. In this study, we present a novel framework combining knowledge graphs (KGs) with auxiliary LLMs to systematically reveal complex bias patterns in medical LLMs. Specifically, the proposed approach integrates adversarial perturbation techniques to identify subtle bias patterns. The approach adopts a customized multi-hop characterization of KGs to enhance the systematic evaluation of arbitrary LLMs. Through a series of comprehensive experiments (on three datasets, six LLMs, and five bias types), we show that our proposed framework has noticeably greater ability and scalability to reveal complex biased patterns of LLMs compared to other baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18993v1">Agent0: Leveraging LLM Agents to Discover Multi-value Features from Text for Enhanced Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Agent4IR, KDD '25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and their associated agent-based frameworks have significantly advanced automated information extraction, a critical component of modern recommender systems. While these multitask frameworks are widely used in code generation, their application in data-centric research is still largely untapped. This paper presents Agent0, an LLM-driven, agent-based system designed to automate information extraction and feature construction from raw, unstructured text. Categorical features are crucial for large-scale recommender systems but are often expensive to acquire. Agent0 coordinates a group of interacting LLM agents to automatically identify the most valuable text aspects for subsequent tasks (such as models or AutoML pipelines). Beyond its feature engineering capabilities, Agent0 also offers an automated prompt-engineering tuning method that utilizes dynamic feedback loops from an oracle. Our findings demonstrate that this closed-loop methodology is both practical and effective for automated feature discovery, which is recognized as one of the most challenging phases in current recommender system development.
    </details>
</div>
