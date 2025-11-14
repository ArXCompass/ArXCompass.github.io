# llm - 2025_06

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
- [Part 13](papers_13.md)
- Part 14
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00250v2">PersianMedQA: Language-Centric Evaluation of LLMs in the Persian Medical Domain</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance on a wide range of NLP benchmarks, often surpassing human-level accuracy. However, their reliability in high-stakes domains such as medicine, particularly in low-resource languages, remains underexplored. In this work, we introduce PersianMedQA, a large-scale, expert-validated dataset of multiple-choice Persian medical questions, designed to evaluate LLMs across both Persian and English. We benchmark over 40 state-of-the-art models, including general-purpose, Persian fine-tuned, and medical LLMs, in zero-shot and chain-of-thought (CoT) settings. Our results show that closed-source general models (e.g., GPT-4.1) consistently outperform all other categories, achieving 83.3% accuracy in Persian and 80.7% in English, while Persian fine-tuned models such as Dorna underperform significantly (e.g., 35.9% in Persian), often struggling with both instruction-following and domain reasoning. We also analyze the impact of translation, showing that while English performance is generally higher, Persian responses are sometimes more accurate due to cultural and clinical contextual cues. Finally, we demonstrate that model size alone is insufficient for robust performance without strong domain or language adaptation. PersianMedQA provides a foundation for evaluating multilingual and culturally grounded medical reasoning in LLMs. The PersianMedQA dataset can be accessed at: https://huggingface.co/datasets/MohammadJRanjbar/PersianMedQA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03444v1">Exploiting LLMs for Automatic Hypothesis Assessment via a Logit-Based Calibrated Prior</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      As hypothesis generation becomes increasingly automated, a new bottleneck has emerged: hypothesis assessment. Modern systems can surface thousands of statistical relationships-correlations, trends, causal links-but offer little guidance on which ones are novel, non-trivial, or worthy of expert attention. In this work, we study the complementary problem to hypothesis generation: automatic hypothesis assessment. Specifically, we ask: given a large set of statistical relationships, can we automatically assess which ones are novel and worth further exploration? We focus on correlations as they are a common entry point in exploratory data analysis that often serve as the basis for forming deeper scientific or causal hypotheses. To support automatic assessment, we propose to leverage the vast knowledge encoded in LLMs' weights to derive a prior distribution over the correlation value of a variable pair. If an LLM's prior expects the correlation value observed, then such correlation is not surprising, and vice versa. We propose the Logit-based Calibrated Prior, an LLM-elicited correlation prior that transforms the model's raw output logits into a calibrated, continuous predictive distribution over correlation values. We evaluate the prior on a benchmark of 2,096 real-world variable pairs and it achieves a sign accuracy of 78.8%, a mean absolute error of 0.26, and 95% credible interval coverage of 89.2% in predicting Pearson correlation coefficient. It also outperforms a fine-tuned RoBERTa classifier in binary correlation prediction and achieves higher precision@K in hypothesis ranking. We further show that the prior generalizes to correlations not seen during LLM pretraining, reflecting context-sensitive reasoning rather than memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03424v1">DistRAG: Towards Distance-Based Spatial Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Many real world tasks where Large Language Models (LLMs) can be used require spatial reasoning, like Point of Interest (POI) recommendation and itinerary planning. However, on their own LLMs lack reliable spatial reasoning capabilities, especially about distances. To address this problem, we develop a novel approach, DistRAG, that enables an LLM to retrieve relevant spatial information not explicitly learned during training. Our method encodes the geodesic distances between cities and towns in a graph and retrieves a context subgraph relevant to the question. Using this technique, our method enables an LLM to answer distance-based reasoning questions that it otherwise cannot answer. Given the vast array of possible places an LLM could be asked about, DistRAG offers a flexible first step towards providing a rudimentary `world model' to complement the linguistic knowledge held in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02165v2">A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 EDM 2025 Long Paper
    </div>
    <details class="paper-abstract">
      Open-ended short-answer questions (SAGs) have been widely recognized as a powerful tool for providing deeper insights into learners' responses in the context of learning analytics (LA). However, SAGs often present challenges in practice due to the high grading workload and concerns about inconsistent assessments. With recent advancements in natural language processing (NLP), automatic short-answer grading (ASAG) offers a promising solution to these challenges. Despite this, current ASAG algorithms are often limited in generalizability and tend to be tailored to specific questions. In this paper, we propose a unified multi-agent ASAG framework, GradeOpt, which leverages large language models (LLMs) as graders for SAGs. More importantly, GradeOpt incorporates two additional LLM-based agents - the reflector and the refiner - into the multi-agent system. This enables GradeOpt to automatically optimize the original grading guidelines by performing self-reflection on its errors. Through experiments on a challenging ASAG task, namely the grading of pedagogical content knowledge (PCK) and content knowledge (CK) questions, GradeOpt demonstrates superior performance in grading accuracy and behavior alignment with human graders compared to representative baselines. Finally, comprehensive ablation studies confirm the effectiveness of the individual components designed in GradeOpt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03396v1">Fault Localisation and Repair for DL Systems: An Empirical Study with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 arXiv admin note: text overlap with arXiv:2301.11568
    </div>
    <details class="paper-abstract">
      Numerous Fault Localisation (FL) and repair techniques have been proposed to address faults in Deep Learning (DL) models. However, their effectiveness in practical applications remains uncertain due to the reliance on pre-defined rules. This paper presents a comprehensive evaluation of state-of-the-art FL and repair techniques, examining their advantages and limitations. Moreover, we introduce a novel approach that harnesses the power of Large Language Models (LLMs) in localising and repairing DL faults. Our evaluation, conducted on a carefully designed benchmark, reveals the strengths and weaknesses of current FL and repair techniques. We emphasise the importance of enhanced accuracy and the need for more rigorous assessment methods that employ multiple ground truth patches. Notably, LLMs exhibit remarkable performance in both FL and repair tasks. For instance, the GPT-4 model achieves 44% and 82% improvements in FL and repair tasks respectively, compared to the second-best tool, demonstrating the potential of LLMs in this domain. Our study sheds light on the current state of FL and repair techniques and suggests that LLMs could be a promising avenue for future advancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05276v2">Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 EDM 2025 Short Paper
    </div>
    <details class="paper-abstract">
      Short answer assessment is a vital component of science education, allowing evaluation of students' complex three-dimensional understanding. Large language models (LLMs) that possess human-like ability in linguistic tasks are increasingly popular in assisting human graders to reduce their workload. However, LLMs' limitations in domain knowledge restrict their understanding in task-specific requirements and hinder their ability to achieve satisfactory performance. Retrieval-augmented generation (RAG) emerges as a promising solution by enabling LLMs to access relevant domain-specific knowledge during assessment. In this work, we propose an adaptive RAG framework for automated grading that dynamically retrieves and incorporates domain-specific knowledge based on the question and student answer context. Our approach combines semantic search and curated educational sources to retrieve valuable reference materials. Experimental results in a science education dataset demonstrate that our system achieves an improvement in grading accuracy compared to baseline LLM approaches. The findings suggest that RAG-enhanced grading systems can serve as reliable support with efficient performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01679v2">VinePPO: Refining Credit Assignment in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted at ICML 2025; 12 pages and 22 pages Appendix
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to complex reasoning tasks that require executing several complex steps before receiving any reward. Properly assigning credit to these steps is essential for enhancing model performance. Proximal Policy Optimization (PPO), a common reinforcement learning (RL) algorithm used for LLM finetuning, employs value networks to tackle credit assignment. However, recent approaches achieve strong results without it, raising questions about the efficacy of value networks in practice. In this work, we systematically evaluate the efficacy of value networks and reveal their significant shortcomings in reasoning-heavy LLM tasks, showing that they often produce poor estimate of expected return and barely outperform a random baseline when comparing alternative steps. This motivates our key question: Can improved credit assignment enhance RL training for LLMs? To address this, we propose VinePPO, a straightforward approach that leverages the flexibility of language environments to compute unbiased Monte Carlo-based estimates. Our method consistently outperforms PPO and other baselines across MATH and GSM8K datasets in less wall-clock time (up to 3.0x). Crucially, it achieves higher test accuracy for a given training accuracy, capturing more generalization signal per sample. These results emphasize the importance of accurate credit assignment in RL training of LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01622v3">DOVE: A Large-Scale Multi-Dimensional Predictions Dataset Towards Meaningful LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Recent work found that LLMs are sensitive to a wide range of arbitrary prompt dimensions, including the type of delimiters, answer enumerators, instruction wording, and more. This throws into question popular single-prompt evaluation practices. We present DOVE (Dataset Of Variation Evaluation) a large-scale dataset containing prompt perturbations of various evaluation benchmarks. In contrast to previous work, we examine LLM sensitivity from an holistic perspective, and assess the joint effects of perturbations along various dimensions, resulting in thousands of perturbations per instance. We evaluate several model families against DOVE, leading to several findings, including efficient methods for choosing well-performing prompts, observing that few-shot examples reduce sensitivity, and identifying instances which are inherently hard across all perturbations. DOVE consists of more than 250M prompt perturbations and model outputs, which we make publicly available to spur a community-wide effort toward meaningful, robust, and efficient evaluation. Browse the data, contribute, and more: https://slab-nlp.github.io/DOVE/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09300v4">Nudging: Inference-time Alignment of LLMs via Guided Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Accepted to ACL 2025 (main)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require alignment to effectively and safely follow user instructions. This process necessitates training an aligned version for every base model, resulting in significant computational overhead. In this work, we propose NUDGING, a simple, training-free algorithm that aligns any base model at inference time using a small aligned model. NUDGING is motivated by recent findings that alignment primarily alters the model's behavior on a small subset of stylistic tokens (e.g., discourse markers). We find that base models are significantly more uncertain when generating these tokens. Building on this insight, NUDGING employs a small aligned model to generate nudging tokens to guide the base model's output during decoding when the base model's uncertainty is high, with only a minor additional inference overhead. We evaluate NUDGING across 3 model families on a diverse range of open-instruction tasks. Without any training, nudging a large base model with a 7x-14x smaller aligned model achieves zero-shot performance comparable to, and sometimes surpassing, that of large aligned models. By operating at the token level, NUDGING enables off-the-shelf collaboration between model families. For instance, nudging Gemma-2-27b with Llama-27b-chat outperforms Llama-2-70b-chat on various tasks. Overall, our work offers a modular and cost-efficient solution to LLM alignment. Our code and demo are available at: https://fywalter.github.io/nudging/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19582v2">Where Are We? Evaluating LLM Performance on African Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Africa's rich linguistic heritage remains underrepresented in NLP, largely due to historical policies that favor foreign languages and create significant data inequities. In this paper, we integrate theoretical insights on Africa's language landscape with an empirical evaluation using Sahara - a comprehensive benchmark curated from large-scale, publicly accessible datasets capturing the continent's linguistic diversity. By systematically assessing the performance of leading large language models (LLMs) on Sahara, we demonstrate how policy-induced data variations directly impact model effectiveness across African languages. Our findings reveal that while a few languages perform reasonably well, many Indigenous languages remain marginalized due to sparse data. Leveraging these insights, we offer actionable recommendations for policy reforms and inclusive data practices. Overall, our work underscores the urgent need for a dual approach - combining theoretical understanding with empirical evaluation - to foster linguistic diversity in AI for African communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20573v2">Collision- and Reachability-Aware Multi-Robot Control with Grounded LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated strong performance in various robot control tasks. However, their deployment in real-world applications remains constrained. Even state-ofthe-art LLMs, such as GPT-o4mini, frequently produce invalid action plans that violate physical constraints, such as directing a robot to an unreachable location or causing collisions between robots. This issue primarily arises from a lack of awareness of these physical constraints during the reasoning process. To address this issue, we propose a novel framework that integrates reinforcement learning with verifiable rewards (RLVR) to incentivize knowledge of physical constraints into LLMs to induce constraints-aware reasoning during plan generation. In this approach, only valid action plans that successfully complete a control task receive positive rewards. We applied our method to two small-scale LLMs: a non-reasoning Qwen2.5-3B-Instruct and a reasoning Qwen3-4B. The experiment results demonstrate that constraint-aware small LLMs largely outperform large-scale models without constraints, grounded on both the BoxNet task and a newly developed BoxNet3D environment built using MuJoCo. This work highlights the effectiveness of grounding even small LLMs with physical constraints to enable scalable and efficient multi-robot control in complex, physically constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03337v1">Mitigating Non-IID Drift in Zeroth-Order Federated LLM Fine-Tuning with Transferable Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 56 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Federated Learning enables collaborative fine-tuning of Large Language Models (LLMs) across decentralized Non-Independent and Identically Distributed (Non-IID) clients, but such models' massive parameter sizes lead to significant memory and communication challenges. This work introduces Meerkat, a sparse zeroth-order optimization (ZO) method designed for federated LLM fine-tuning. By limiting fine-tuning to a transferable, static, extremely sparse subset of parameters, Meerkat achieves remarkable communication efficiency, enabling cost-effective high-frequency synchronization. With theoretical analysis and experiments, we show that this high-frequency communication effectively mitigates Non-IID data challenges and leads to superior performance compared to full-parameter ZO. Furthermore, experiment results show that Meerkat outperforms existing sparsity baselines with better performance at the same communication frequency. To further handle Non-IID drift, Meerkat leverages traceable local updates and forms a virtual path for each client. This virtual path mechanism reveals the GradIP phenomenon: the inner products between LLM pre-training gradients maintained by server and client gradients estimated via ZO converges for extreme Non-IID clients but oscillates for IID ones. This distinct behavior provides a signal for identifying clients with extreme data heterogeneity. Using this signal, Meerkat-vp is proposed to analyze GradIP trajectories to identify extreme Non-IID clients and applies early stopping to enhance aggregated model quality. Experiments confirm that Meerkat and Meerkat-vp significantly improve the efficiency and effectiveness of ZO federated LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03296v1">Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) for online inference is often constrained by limited GPU memory, particularly due to the growing KV cache during auto-regressive decoding. Hybrid GPU-CPU execution has emerged as a promising solution by offloading KV cache management and parts of attention computation to the CPU. However, a key bottleneck remains: existing schedulers fail to effectively overlap CPU-offloaded tasks with GPU execution during the latency-critical, bandwidth-bound decode phase. This particularly penalizes real-time, decode-heavy applications (e.g., chat, Chain-of-Thought reasoning) which are currently underserved by existing systems, especially under memory pressure typical of edge or low-cost deployments. We present APEX, a novel, profiling-informed scheduling strategy that maximizes CPU-GPU parallelism during hybrid LLM inference. Unlike systems relying on static rules or purely heuristic approaches, APEX dynamically dispatches compute across heterogeneous resources by predicting execution times of CPU and GPU subtasks to maximize overlap while avoiding scheduling overheads.We evaluate APEX on diverse workloads and GPU architectures (NVIDIA T4, A10), using LLaMa-2-7B and LLaMa-3.1-8B models. Compared to GPU-only schedulers like VLLM, APEX improves throughput by 84% - 96% on T4 and 11% - 89% on A10 GPUs, while preserving latency. Against the best existing hybrid schedulers, it delivers up to 49% (T4) and 37% (A10) higher throughput in long-output settings.APEX significantly advances hybrid LLM inference efficiency on such memory-constrained hardware and provides a blueprint for scheduling in heterogeneous AI systems, filling a critical gap for efficient real-time LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03295v1">Unleashing the Reasoning Potential of Pre-trained LLMs by Critique Fine-Tuning on One Problem</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We have witnessed that strong LLMs like Qwen-Math, MiMo, and Phi-4 possess immense reasoning potential inherited from the pre-training stage. With reinforcement learning (RL), these models can improve dramatically on reasoning tasks. Recent studies have shown that even RL on a single problem can unleash these models' reasoning capabilities. However, RL is not only expensive but also unstable. Even one-shot RL requires hundreds of GPU hours. This raises a critical question: Is there a more efficient way to unleash the reasoning potential of these powerful base LLMs? In this work, we demonstrate that Critique Fine-Tuning (CFT) on only one problem can effectively unleash the reasoning potential of LLMs. Our method constructs critique data by collecting diverse model-generated solutions to a single problem and using teacher LLMs to provide detailed critiques. We fine-tune Qwen and Llama family models, ranging from 1.5B to 14B parameters, on the CFT data and observe significant performance gains across diverse reasoning tasks. For example, with just 5 GPU hours of training, Qwen-Math-7B-CFT show an average improvement of 15% on six math benchmarks and 16% on three logic reasoning benchmarks. These results are comparable to or even surpass the results from RL with 20x less compute. Ablation studies reveal the robustness of one-shot CFT across different prompt problems. These results highlight one-shot CFT as a simple, general, and compute-efficient approach to unleashing the reasoning capabilities of modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03231v1">NetPress: Dynamically Generated LLM Benchmarks for Network Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Despite growing interest in domain-specific benchmarking of large language models (LLMs) and agents, current evaluations remain limited to static, small-scale datasets, especially in high-stakes tasks like network operations that demand reliability for deployments. We present NetPress, an automated benchmark generation framework for evaluating LLM agents in network applications. NetPress introduces a unified abstraction with state and action, enabling dynamic generation of diverse query sets along with corresponding ground truths. At runtime, users can specify benchmark configurations to generate millions of queries on the fly. In addition to dynamic benchmark construction, NetPress integrates with network emulators to provide realistic environment feedback, supporting comprehensive evaluation across correctness, safety, and latency. We instantiate NetPress on three representative applications, revealing interesting fine-grained differences in agent behavior that static, correctness-only benchmarks often miss. NetPress moves LLM evaluation toward realistic, scalable testing in infrastructure-centric domains, helping close the gap between benchmark performance and real-world deployment readiness. Code is available at https://github.com/Froot-NetSys/NetPress.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03145v1">Entity-Augmented Neuroscience Knowledge Retrieval Using Ontology and Semantic Understanding Capability of LLM</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Neuroscience research publications encompass a vast wealth of knowledge. Accurately retrieving existing information and discovering new insights from this extensive literature is essential for advancing the field. However, when knowledge is dispersed across multiple sources, current state-of-the-art retrieval methods often struggle to extract the necessary information. A knowledge graph (KG) can integrate and link knowledge from multiple sources, but existing methods for constructing KGs in neuroscience often rely on labeled data and require domain expertise. Acquiring large-scale, labeled data for a specialized area like neuroscience presents significant challenges. This work proposes novel methods for constructing KG from unlabeled large-scale neuroscience research corpus utilizing large language models (LLM), neuroscience ontology, and text embeddings. We analyze the semantic relevance of neuroscience text segments identified by LLM for building the knowledge graph. We also introduce an entity-augmented information retrieval algorithm to extract knowledge from the KG. Several experiments were conducted to evaluate the proposed approaches, and the results demonstrate that our methods significantly enhance knowledge discovery from the unlabeled neuroscience research corpus. It achieves an F1 score of 0.84 for entity extraction, and the knowledge obtained from the KG improves answers to over 54% of the questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03139v1">SVGenius: Benchmarking LLMs in SVG Understanding, Editing and Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 19 pages,4 figures, Project page: https://zju-real.github.io/SVGenius, Code: https://github.com/ZJU-REAL/SVGenius-Bench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Multimodal LLMs have shown promising capabilities for SVG processing, yet existing benchmarks suffer from limited real-world coverage, lack of complexity stratification, and fragmented evaluation paradigms. We introduce SVGenius, a comprehensive benchmark comprising 2,377 queries across three progressive dimensions: understanding, editing, and generation. Built on real-world data from 24 application domains with systematic complexity stratification, SVGenius evaluates models through 8 task categories and 18 metrics. We assess 22 mainstream models spanning different scales, architectures, training paradigms, and accessibility levels. Our analysis reveals that while proprietary models significantly outperform open-source counterparts, all models exhibit systematic performance degradation with increasing complexity, indicating fundamental limitations in current approaches; however, reasoning-enhanced training proves more effective than pure scaling for overcoming these limitations, though style transfer remains the most challenging capability across all model types. SVGenius establishes the first systematic evaluation framework for SVG processing, providing crucial insights for developing more capable vector graphics models and advancing automated graphic design applications. Appendix and supplementary materials (including all data and code) are available at https://zju-real.github.io/SVGenius.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03122v1">AUTOCIRCUIT-RL: Reinforcement Learning-Driven LLM for Automated Circuit Topology Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 9 Pages (Content), 4 Pages (Appendix), 7 figures, ICML'2025
    </div>
    <details class="paper-abstract">
      Analog circuit topology synthesis is integral to Electronic Design Automation (EDA), enabling the automated creation of circuit structures tailored to specific design requirements. However, the vast design search space and strict constraint adherence make efficient synthesis challenging. Leveraging the versatility of Large Language Models (LLMs), we propose AUTOCIRCUIT-RL,a novel reinforcement learning (RL)-based framework for automated analog circuit synthesis. The framework operates in two phases: instruction tuning, where an LLM learns to generate circuit topologies from structured prompts encoding design constraints, and RL refinement, which further improves the instruction-tuned model using reward models that evaluate validity, efficiency, and output voltage. The refined model is then used directly to generate topologies that satisfy the design constraints. Empirical results show that AUTOCIRCUIT-RL generates ~12% more valid circuits and improves efficiency by ~14% compared to the best baselines, while reducing duplicate generation rates by ~38%. It achieves over 60% success in synthesizing valid circuits with limited training data, demonstrating strong generalization. These findings highlight the framework's effectiveness in scaling to complex circuits while maintaining efficiency and constraint adherence, marking a significant advancement in AI-driven circuit design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v1">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 38 pages
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03095v1">DPO Learning with LLMs-Judge Signal for Computer Use Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Computer use agents (CUA) are systems that automatically interact with graphical user interfaces (GUIs) to complete tasks. CUA have made significant progress with the advent of large vision-language models (VLMs). However, these agents typically rely on cloud-based inference with substantial compute demands, raising critical privacy and scalability concerns, especially when operating on personal devices. In this work, we take a step toward privacy-preserving and resource-efficient agents by developing a lightweight vision-language model that runs entirely on local machines. To train this compact agent, we introduce an LLM-as-Judge framework that automatically evaluates and filters synthetic interaction trajectories, producing high-quality data for reinforcement learning without human annotation. Experiments on the OS-World benchmark demonstrate that our fine-tuned local model outperforms existing baselines, highlighting a promising path toward private, efficient, and generalizable GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13172v2">Unveiling Privacy Risks in LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have become increasingly prevalent across various real-world applications. They enhance decision-making by storing private user-agent interactions in the memory module for demonstrations, introducing new privacy risks for LLM agents. In this work, we systematically investigate the vulnerability of LLM agents to our proposed Memory EXTRaction Attack (MEXTRA) under a black-box setting. To extract private information from memory, we propose an effective attacking prompt design and an automated prompt generation method based on different levels of knowledge about the LLM agent. Experiments on two representative agents demonstrate the effectiveness of MEXTRA. Moreover, we explore key factors influencing memory leakage from both the agent designer's and the attacker's perspectives. Our findings highlight the urgent need for effective memory safeguards in LLM agent design and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03051v1">Facts Do Care About Your Language: Assessing Answer Quality of Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Factuality is a necessary precursor to useful educational tools. As adoption of Large Language Models (LLMs) in education continues of grow, ensuring correctness in all settings is paramount. Despite their strong English capabilities, LLM performance in other languages is largely untested. In this work, we evaluate the correctness of the Llama3.1 family of models in answering factual questions appropriate for middle and high school students. We demonstrate that LLMs not only provide extraneous and less truthful information, but also exacerbate existing biases against rare languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02965v1">Memory-Efficient and Privacy-Preserving Collaborative Training for Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 20 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) has been gaining popularity due to its successful adaptation to large language models (LLMs). In this work, we introduce Privacy-preserving Collaborative Mixture-of-Experts (PC-MoE), which leverages the sparsity of the MoE architecture for memory-efficient decentralized collaborative LLM training, enabling multiple parties with limited GPU-memory and data resources to collectively train more capable LLMs than they could achieve individually. At the same time, this approach protects training data privacy of each participant by keeping training data, as well as parts of the forward pass signal and gradients locally within each party. By design, PC-MoE synergistically combines the strengths of distributed computation with strong confidentiality assurances. Unlike most privacy-preserving schemes, which pay for confidentiality with lower task accuracy, our framework breaks that trade-off: across seven popular LLM benchmarks, it almost matches (and sometimes exceeds) the performance and convergence rate of a fully centralized model, enjoys near 70% peak GPU RAM reduction, while being fully robust against reconstruction attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02954v1">Towards More Effective Fault Detection in LLM-Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Unit tests play a vital role in uncovering potential faults in software. While tools like EvoSuite focus on maximizing code coverage, recent advances in large language models (LLMs) have shifted attention toward LLM-based test generation. However, code coverage metrics -- such as line and branch coverage -- remain overly emphasized in reported research, despite being weak indicators of a test suite's fault-detection capability. In contrast, \textit{mutation score} offers a more reliable and stringent measure, as demonstrated in our findings where some test suites achieve 100\% coverage but only 4\% mutation score. Although a few studies consider mutation score, the effectiveness of LLMs in killing mutants remains underexplored. In this paper, we propose MUTGEN, a mutation-guided, LLM-based test generation approach that incorporates mutation feedback directly into the prompt. Evaluated on 204 subjects from two benchmarks, MUTGEN significantly outperforms both EvoSuite and vanilla prompt-based strategies in terms of mutation score. Furthermore, MUTGEN introduces an iterative generation mechanism that pushes the limits of LLMs in killing additional mutants. Our study also provide insights into the limitations of LLM-based generation, analyzing the reasons for live and uncovered mutants, and the impact of different mutation operators on generation effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02862v2">Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02945v1">Quantitative LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge is a framework in which a large language model (LLM) automatically evaluates the output of another LLM. We propose quantitative LLM judges, which align evaluation scores of existing LLM judges to human scores in a given domain using regression models. The models are trained to improve the score of the original judge by using the judge's textual evaluation and score. We present four quantitative judges for different types of absolute and relative feedback, which showcases the generality and versatility of our framework. Our framework is more computationally efficient than supervised fine-tuning and can be more statistically efficient when human feedback is limited, which is expected in most applications of our work. We validate these claims empirically on four datasets using two base judges. Our experiments show that quantitative judges can effectively improve the predictive power of existing judges through post-hoc modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v1">A Multi-agent LLM-based JUit Test Generation with Strong Oracles</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is laborious, especially for strong typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to generate tests that achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in large language models (LLMs) have enabled oracle generation from natural language descriptions. However, existing LLM-based methods often require LLM fine-tuning or rely on external tools such as EvoSuite for test prefix generation. In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM framework for automated JUnit test generation. CANDOR orchestrates multiple specialized LLM agents to generate JUnit tests, including both high-quality test prefixes and accurate oracles. To mitigate the notorious hallucinations in LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generate accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments on the HumanEvalJava and LeetCodeJava datasets show that CANDOR can generate accurate oracles and is slightly better than EvoSuite in generating tests with high line coverage and clearly superior in terms of mutation score. Moreover, CANDOR significantly outperforms the state-of-the-art, prompt-based test generator LLM-Empirical, achieving improvements of 15.8 to 25.1 percentage points in oracle correctness on both correct and faulty source code. Ablation studies confirm the critical contributions of key agents in improving test prefix quality and oracle accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02940v1">Memory-Efficient Split Federated Learning for LLM Fine-Tuning on Heterogeneous Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 IEEE INFOCOM IEILM 2025
    </div>
    <details class="paper-abstract">
      In this paper, we propose an edge-assisted split federated learning framework to facilitate large language model (LLM) fine-tuning on heterogeneous mobile devices while alleviating memory pressures on both mobile devices and the edge server. Specifically, mobile devices perform low-rank adaptation (LoRA) fine-tuning on only a subset of lower layers of the pre-trained LLM, tailored to their individual capacities. On the server, a full LLM is maintained, and the corresponding LoRA modules are selectively fine-tuned in a sequential manner for each device. To further enhance training efficiency, we propose a server-side training scheduling method that optimizes the processing order of devices for accelerating fine-tuning. Extensive experiments demonstrate that compared to the baselines, our scheme can reduce 79\% memory footprint and 6\% training time while achieving comparable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02918v1">Sample, Predict, then Proceed: Self-Verification Sampling for Tool Use of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      Tool use in stateful environments presents unique challenges for large language models (LLMs), where existing test-time compute strategies relying on repeated trials in the environment are impractical. We propose dynamics modelling (DyMo), a method that augments LLMs with a state prediction capability alongside function calling during post-training. This enables LLMs to predict the future states of their actions through an internal environment model. On the Berkeley Function Calling Leaderboard V2, DyMo improves success rates and significantly reduces hallucinations. We further integrate the internal environment model into self-verification sampling (SVS), and show that this substantially improves pass^k over number of trials k, and allows the model to refuse unreliable outputs. Together, DyMo and SVS greatly enhance the effectiveness and reliability of LLMs for tool use. We believe this work charts a path towards scalable planning RL methods for LLM inference without repeatedly querying the oracle environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02911v1">Cell-o1: Training LLMs to Solve Single-Cell Reasoning Puzzles with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
      | 💬 28 pages; 16 tables; 7 figures; Code: https://github.com/ncbi-nlp/cell-o1
    </div>
    <details class="paper-abstract">
      Cell type annotation is a key task in analyzing the heterogeneity of single-cell RNA sequencing data. Although recent foundation models automate this process, they typically annotate cells independently, without considering batch-level cellular context or providing explanatory reasoning. In contrast, human experts often annotate distinct cell types for different cell clusters based on their domain knowledge. To mimic this workflow, we introduce the CellPuzzles task, where the objective is to assign unique cell types to a batch of cells. This benchmark spans diverse tissues, diseases, and donor conditions, and requires reasoning across the batch-level cellular context to ensure label uniqueness. We find that off-the-shelf large language models (LLMs) struggle on CellPuzzles, with the best baseline (OpenAI's o1) achieving only 19.0% batch-level accuracy. To fill this gap, we propose Cell-o1, a 7B LLM trained via supervised fine-tuning on distilled reasoning traces, followed by reinforcement learning with batch-level rewards. Cell-o1 achieves state-of-the-art performance, outperforming o1 by over 73% and generalizing well across contexts. Further analysis of training dynamics and reasoning behaviors provides insights into batch-level annotation performance and emergent expert-like reasoning. Code and data are available at https://github.com/ncbi-nlp/cell-o1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18924v2">LLM-Guided Taxonomy and Hierarchical Uncertainty for 3D Point Cloud Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-03
    </div>
    <details class="paper-abstract">
      We present a novel active learning framework for 3D point cloud semantic segmentation that, for the first time, integrates large language models (LLMs) to construct hierarchical label structures and guide uncertainty-based sample selection. Unlike prior methods that treat labels as flat and independent, our approach leverages LLM prompting to automatically generate multi-level semantic taxonomies and introduces a recursive uncertainty projection mechanism that propagates uncertainty across hierarchy levels. This enables spatially diverse, label-aware point selection that respects the inherent semantic structure of 3D scenes. Experiments on S3DIS and ScanNet v2 show that our method achieves up to 4% mIoU improvement under extremely low annotation budgets (e.g., 0.02%), substantially outperforming existing baselines. Our results highlight the untapped potential of LLMs as knowledge priors in 3D vision and establish hierarchical uncertainty modeling as a powerful paradigm for efficient point cloud annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02314v1">ResearchCodeBench: Benchmarking LLMs on Implementing Novel Machine Learning Research Code</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in transforming machine learning research, yet their capability to faithfully implement novel ideas from recent research papers-ideas unseen during pretraining-remains unclear. We introduce ResearchCodeBench, a benchmark of 212 coding challenges that evaluates LLMs' ability to translate cutting-edge ML contributions from top 2024-2025 research papers into executable code. We assessed 30+ proprietary and open-source LLMs, finding that even the best models correctly implement less than 40% of the code. We find Gemini-2.5-Pro-Preview to perform best at 37.3% success rate, with O3 (High) and O4-mini (High) following behind at 32.3% and 30.8% respectively. We present empirical findings on performance comparison, contamination, and error patterns. By providing a rigorous and community-driven evaluation platform, ResearchCodeBench enables continuous understanding and advancement of LLM-driven innovation in research code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v9">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01042v4">SafeSwitch: Steering Unsafe LLM Behavior via Internal Activation Signals</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit exceptional capabilities across various tasks but also pose risks by generating harmful content. Existing safety mechanisms, while improving model safety, often lead to overly cautious behavior and fail to fully leverage LLMs' internal cognitive processes. Inspired by humans' reflective thinking capability, we first show that LLMs can similarly perform internal assessments about safety in their internal states. Building on this insight, we propose SafeSwitch, a dynamic framework that regulates unsafe outputs by utilizing the prober-based internal state monitor that actively detects harmful intentions, and activates a safety head that leads to safer and more conservative responses only when necessary. SafeSwitch reduces harmful outputs by approximately 80% on harmful queries while maintaining strong utility, reaching a Pareto optimal among several methods. Our method is also advantageous over traditional methods in offering more informative, context-aware refusals, and achieves these benefits while only tuning less than 6% of the original parameters. SafeSwitch demonstrates large language models' capacity for self-awareness and reflection regarding safety, offering a promising approach to more nuanced and effective safety controls. Codes for this work are available at https://github.com/Hanpx20/SafeSwitch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19501v2">Toward Scientific Reasoning in LLMs: Training from Expert Discussions via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We investigate how to teach large language models (LLMs) to perform scientific reasoning by leveraging expert discussions as a learning signal. Focusing on the genomics domain, we develop an automated pipeline to extract trainable data and introduce Genome-Bench, a new benchmark constructed from over a decade of scientific forum discussions on genome engineering. Our pipeline transforms raw interactions into a reinforcement learning-friendly multiple-choice questions format, supported by 3000+ high-quality question-answer pairs spanning foundational biology, experimental troubleshooting, tool usage, and beyond. We fine-tune an LLM using RL with a rule-based reward signal derived from the synthetic MCQ dataset to enhance domain-specific reasoning. Our results show that reinforcement learning from scientific discussions improves model performance by over 15% compared to the base model on Genome-Bench, narrowing the gap between open-source LLMs and expert-level reasoning. To our knowledge, this is the first end-to-end pipeline for teaching LLMs to reason from scientific discussions, with promising potential for generalization across scientific domains beyond biology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17579v2">Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between pronoun production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size-with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior. Our codes and results are available at https://github.com/LingMechLab/Production-Interpretation_Asymmetries_ACL2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02211v1">Improving LLM-Generated Code Quality with GRPO</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are gaining widespread use for code generation. Recent training procedures use execution feedback as a reward signal, typically focusing on the functional correctness of the code, using unit test pass rate as a reward signal. However, this reward signal fails to capture notions of maintainability, quality and safety of the code produced. We address this under-explored area and develop a comprehensive library to quantify various aspects of code quality, and use it as a reward in GRPO. We find GRPO increases code quality according to this measure, which is confirmed by expert, blinded human annotators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02208v1">KDRL: Post-Training Reasoning LLMs via Unified Knowledge Distillation and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) post-training have leveraged two distinct paradigms to enhance reasoning capabilities: reinforcement learning (RL) and knowledge distillation (KD). While RL enables the emergence of complex reasoning behaviors, it often suffers from low sample efficiency when the initial policy struggles to explore high-reward trajectories. Conversely, KD improves learning efficiency via mimicking the teacher model but tends to generalize poorly to out-of-domain scenarios. In this work, we present \textbf{KDRL}, a \textit{unified post-training framework} that jointly optimizes a reasoning model through teacher supervision (KD) and self-exploration (RL). Specifically, KDRL leverages policy gradient optimization to simultaneously minimize the reverse Kullback-Leibler divergence (RKL) between the student and teacher distributions while maximizing the expected rule-based rewards. We first formulate a unified objective that integrates GRPO and KD, and systematically explore how different KL approximations, KL coefficients, and reward-guided KD strategies affect the overall post-training dynamics and performance. Empirical results on multiple reasoning benchmarks demonstrate that KDRL outperforms GRPO and various KD baselines while achieving a favorable balance between performance and reasoning token efficiency. These findings indicate that integrating KD and RL serves as an effective and efficient strategy to train reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03525v4">UnSeenTimeQA: Time-Sensitive Question-Answering Beyond LLMs' Memorization</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      This paper introduces UnSeenTimeQA, a novel data contamination-free time-sensitive question-answering (TSQA) benchmark. It differs from existing TSQA benchmarks by avoiding web-searchable queries grounded in the real world. We present a series of time-sensitive event scenarios based on synthetically generated facts. It requires large language models (LLMs) to engage in genuine temporal reasoning without depending on the factual knowledge acquired during the pre-training phase. Our data generation framework enables on-demand generation of new samples, mitigating the risk of data leakage. We designed three types of time-sensitive questions to test LLMs' temporal reasoning abilities over sequential and parallel event occurrences. Our evaluation of five LLMs on synthetic fact-based TSQA reveals mixed results: while they perform well on simpler subsets, their overall performance remains inferior as compared to real world fact-based TSQA. Error analysis indicates that LLMs face difficulties in reasoning over long-range event dependencies and parallel events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02177v1">Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning, such as PPO and GRPO, has powered recent breakthroughs in LLM reasoning. Scaling rollout to sample more prompts enables models to selectively use higher-quality data for training, which can stabilize RL training and improve model performance. However, this comes at the cost of significant computational overhead. In this paper, we show that a substantial portion of this overhead can be avoided by skipping uninformative prompts before rollout. Our analysis of reward dynamics reveals a strong temporal consistency in prompt value: prompts that are uninformative in one epoch of training are likely to remain uninformative in future epochs. Based on these insights, we propose GRESO (GRPO with Efficient Selective Rollout), an online, lightweight pre-rollout filtering algorithm that predicts and skips uninformative prompts using reward training dynamics. By evaluating GRESO on a broad range of math reasoning benchmarks and models, such as Qwen2.5-Math-1.5B, DeepSeek-R1-Distill-Qwen-1.5B, and Qwen2.5-Math-7B, we show that GRESO achieves up to 2.4x wall-clock time speedup in rollout and up to 2.0x speedup in total training time without accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16873v2">AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to ICML 2025. Code is available at http://github.com/facebookresearch/advprompter
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to jailbreaking attacks that lead to generation of inappropriate or harmful content. Manual red-teaming requires a time-consuming search for adversarial prompts, whereas automatic adversarial prompt generation often leads to semantically meaningless attacks that do not scale well. In this paper, we present a novel method that uses another LLM, called AdvPrompter, to generate human-readable adversarial prompts in seconds. AdvPrompter, which is trained using an alternating optimization algorithm, generates suffixes that veil the input instruction without changing its meaning, such that the TargetLLM is lured to give a harmful response. Experimental results on popular open source TargetLLMs show highly competitive results on the AdvBench and HarmBench datasets, that also transfer to closed-source black-box LLMs. We also show that training on adversarial suffixes generated by AdvPrompter is a promising strategy for improving the robustness of LLMs to jailbreaking attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06150v2">A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01954v1">DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 Main. Code is available at https://github.com/VILA-Lab/DRAG
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) methods have proven highly effective for tasks requiring factual consistency and robust knowledge retrieval. However, large-scale RAG systems consume significant computational resources and are prone to generating hallucinated content from Humans. In this work, we introduce $\texttt{DRAG}$, a novel framework for distilling RAG knowledge from large-scale Language Models (LLMs) into small LMs (SLMs). Our approach leverages evidence- and knowledge graph-based distillation, ensuring that the distilled model retains critical factual knowledge while significantly reducing model size and computational cost. By aligning the smaller model's predictions with a structured knowledge graph and ranked evidence, $\texttt{DRAG}$ effectively mitigates hallucinations and improves factual accuracy. We further present a case demonstrating how our framework mitigates user privacy risks and introduce a corresponding benchmark. Experimental evaluations on multiple benchmarks demonstrate that our method outperforms the prior competitive RAG methods like MiniRAG for SLMs by up to 27.7% using the same models, preserving high-level efficiency and reliability. With $\texttt{DRAG}$, we provide a practical and resource-efficient roadmap to deploying enhanced retrieval and generation capabilities in small-sized LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01939v1">Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 25 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01900v1">COALESCE: Economic and Security Dynamics of Skill-Based Task Outsourcing Among Team of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 2 figures, github linked
    </div>
    <details class="paper-abstract">
      The meteoric rise and proliferation of autonomous Large Language Model (LLM) agents promise significant capabilities across various domains. However, their deployment is increasingly constrained by substantial computational demands, specifically for Graphics Processing Unit (GPU) resources. This paper addresses the critical problem of optimizing resource utilization in LLM agent systems. We introduce COALESCE (Cost-Optimized and Secure Agent Labour Exchange via Skill-based Competence Estimation), a novel framework designed to enable autonomous LLM agents to dynamically outsource specific subtasks to specialized, cost-effective third-party LLM agents. The framework integrates mechanisms for hybrid skill representation, dynamic skill discovery, automated task decomposition, a unified cost model comparing internal execution costs against external outsourcing prices, simplified market-based decision-making algorithms, and a standardized communication protocol between LLM agents. Comprehensive validation through 239 theoretical simulations demonstrates 41.8\% cost reduction potential, while large-scale empirical validation across 240 real LLM tasks confirms 20.3\% cost reduction with proper epsilon-greedy exploration, establishing both theoretical viability and practical effectiveness. The emergence of proposed open standards like Google's Agent2Agent (A2A) protocol further underscores the need for frameworks like COALESCE that can leverage such standards for efficient agent interaction. By facilitating a dynamic market for agent capabilities, potentially utilizing protocols like A2A for communication, COALESCE aims to significantly reduce operational costs, enhance system scalability, and foster the emergence of specialized agent economies, making complex LLM agent functionalities more accessible and economically viable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03505v2">Optimus: Accelerating Large-Scale Multi-Modal LLM Training by Bubble Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have extended the success of large language models (LLMs) to multiple data types, such as image, text and audio, achieving significant performance in various domains, including multimodal translation, visual question answering and content generation. Nonetheless, existing systems are inefficient to train MLLMs due to substantial GPU bubbles caused by the heterogeneous modality models and complex data dependencies in 3D parallelism. This paper proposes Optimus, a distributed MLLM training system that reduces end-to-end MLLM training time. Optimus is based on our principled analysis that scheduling the encoder computation within the LLM bubbles can reduce bubbles in MLLM training. To make scheduling encoder computation possible for all GPUs, Optimus searches the separate parallel plans for encoder and LLM, and adopts a bubble scheduling algorithm to enable exploiting LLM bubbles without breaking the original data dependencies in the MLLM model architecture. We further decompose encoder layer computation into a series of kernels, and analyze the common bubble pattern of 3D parallelism to carefully optimize the sub-millisecond bubble scheduling, minimizing the overall training time. Our experiments in a production cluster show that Optimus accelerates MLLM training by 20.5%-21.3% with ViT-22B and GPT-175B model over 3072 GPUs compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10918v2">INVARLLM: LLM-assisted Physical Invariant Extraction for Cyber-Physical Systems Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Cyber-Physical Systems (CPS) are vulnerable to cyber-physical attacks that violate physical laws. While invariant-based anomaly detection is effective, existing methods are limited: data-driven approaches lack semantic context, and physics-based models require extensive manual work. We propose INVARLLM, a hybrid framework that uses large language models (LLMs) to extract semantic information from CPS documentation and generate physical invariants, then validates these against real system data using a PCMCI+-inspired K-means method. This approach combines LLM semantic understanding with empirical validation to ensure both interpretability and reliability. We evaluate INVARLLM on SWaT and WADI datasets, achieving 100% precision in anomaly detection with no false alarms, outperforming all existing methods. Our results demonstrate that integrating LLM-derived semantics with statistical validation provides a scalable and dependable solution for CPS security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01839v1">Beyond Static Responses: Multi-Agent LLM Systems as a New Paradigm for Social Science Research</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) transition from static tools to fully agentic systems, their potential for transforming social science research has become increasingly evident. This paper introduces a structured framework for understanding the diverse applications of LLM-based agents, ranging from simple data processors to complex, multi-agent systems capable of simulating emergent social dynamics. By mapping this developmental continuum across six levels, the paper clarifies the technical and methodological boundaries between different agentic architectures, providing a comprehensive overview of current capabilities and future potential. It highlights how lower-tier systems streamline conventional tasks like text classification and data annotation, while higher-tier systems enable novel forms of inquiry, including the study of group dynamics, norm formation, and large-scale social processes. However, these advancements also introduce significant challenges, including issues of reproducibility, ethical oversight, and the risk of emergent biases. The paper critically examines these concerns, emphasizing the need for robust validation protocols, interdisciplinary collaboration, and standardized evaluation metrics. It argues that while LLM-based agents hold transformative potential for the social sciences, realizing this promise will require careful, context-sensitive deployment and ongoing methodological refinement. The paper concludes with a call for future research that balances technical innovation with ethical responsibility, encouraging the development of agentic systems that not only replicate but also extend the frontiers of social science, offering new insights into the complexities of human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01825v1">Which Factors Make Code LLMs More Vulnerable to Backdoor Attacks? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Code LLMs are increasingly employed in software development. However, studies have shown that they are vulnerable to backdoor attacks: when a trigger (a specific input pattern) appears in the input, the backdoor will be activated and cause the model to generate malicious outputs. Researchers have designed various triggers and demonstrated the feasibility of implanting backdoors by poisoning a fraction of the training data. Some basic conclusions have been made, such as backdoors becoming easier to implant when more training data are modified. However, existing research has not explored other factors influencing backdoor attacks on Code LLMs, such as training batch size, epoch number, and the broader design space for triggers, e.g., trigger length. To bridge this gap, we use code summarization as an example to perform an empirical study that systematically investigates the factors affecting backdoor effectiveness and understands the extent of the threat posed. Three categories of factors are considered: data, model, and inference, revealing previously overlooked findings. We find that the prevailing consensus -- that attacks are ineffective at extremely low poisoning rates -- is incorrect. The absolute number of poisoned samples matters as well. Specifically, poisoning just 20 out of 454K samples (0.004\% poisoning rate -- far below the minimum setting of 0.1\% in prior studies) successfully implants backdoors! Moreover, the common defense is incapable of removing even a single poisoned sample from it. Additionally, small batch sizes increase the risk of backdoor attacks. We also uncover other critical factors such as trigger types, trigger length, and the rarity of tokens in the triggers, leading to valuable insights for assessing Code LLMs' vulnerability to backdoor attacks. Our study highlights the urgent need for defense mechanisms against extremely low poisoning rate settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01814v1">Analysis of LLM Bias (Chinese Propaganda & Anti-US Sentiment) in DeepSeek-R1 vs. ChatGPT o3-mini-high</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly shape public understanding and civic decisions, yet their ideological neutrality is a growing concern. While existing research has explored various forms of LLM bias, a direct, cross-lingual comparison of models with differing geopolitical alignments-specifically a PRC-system model versus a non-PRC counterpart-has been lacking. This study addresses this gap by systematically evaluating DeepSeek-R1 (PRC-aligned) against ChatGPT o3-mini-high (non-PRC) for Chinese-state propaganda and anti-U.S. sentiment. We developed a novel corpus of 1,200 de-contextualized, reasoning-oriented questions derived from Chinese-language news, presented in Simplified Chinese, Traditional Chinese, and English. Answers from both models (7,200 total) were assessed using a hybrid evaluation pipeline combining rubric-guided GPT-4o scoring with human annotation. Our findings reveal significant model-level and language-dependent biases. DeepSeek-R1 consistently exhibited substantially higher proportions of both propaganda and anti-U.S. bias compared to ChatGPT o3-mini-high, which remained largely free of anti-U.S. sentiment and showed lower propaganda levels. For DeepSeek-R1, Simplified Chinese queries elicited the highest bias rates; these diminished in Traditional Chinese and were nearly absent in English. Notably, DeepSeek-R1 occasionally responded in Simplified Chinese to Traditional Chinese queries and amplified existing PRC-aligned terms in its Chinese answers, demonstrating an "invisible loudspeaker" effect. Furthermore, such biases were not confined to overtly political topics but also permeated cultural and lifestyle content, particularly in DeepSeek-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01804v1">A Study on the MCP x A2A Framework for Enhancing Interoperability of LLM-based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper provides an in-depth technical analysis and implementation methodology of the open-source Agent-to-Agent (A2A) protocol developed by Google and the Model Context Protocol (MCP) introduced by Anthropic. While the evolution of LLM-based autonomous agents is rapidly accelerating, efficient interactions among these agents and their integration with external systems remain significant challenges. In modern AI systems, collaboration between autonomous agents and integration with external tools have become essential elements for building practical AI applications. A2A offers a standardized communication method that enables agents developed in heterogeneous environments to collaborate effectively, while MCP provides a structured I/O framework for agents to connect with external tools and resources. Prior studies have focused primarily on the features and applications of either A2A or MCP individually. In contrast, this study takes an integrated approach, exploring how the two protocols can complement each other to address interoperability issues and facilitate efficient collaboration within complex agent ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01770v1">ReGA: Representation-Guided Abstraction for Model-based Safeguarding of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in various tasks, yet concerns about their safety and security have emerged. In particular, they pose risks in generating harmful content and vulnerability to jailbreaking attacks. To analyze and monitor machine learning models, model-based analysis has demonstrated notable potential in stateful deep neural networks, yet suffers from scalability issues when extending to LLMs due to their vast feature spaces. In this paper, we propose ReGA, a model-based analysis framework with representation-guided abstraction, to safeguard LLMs against harmful prompts and generations. By leveraging safety-critical representations, which are low-dimensional directions emerging in hidden states that indicate safety-related concepts, ReGA effectively addresses the scalability issue when constructing the abstract model for safety modeling. Our comprehensive evaluation shows that ReGA performs sufficiently well in distinguishing between safe and harmful inputs, achieving an AUROC of 0.975 at the prompt level and 0.985 at the conversation level. Additionally, ReGA exhibits robustness to real-world attacks and generalization across different safety perspectives, outperforming existing safeguard paradigms in terms of interpretability and scalability. Overall, ReGA serves as an efficient and scalable solution to enhance LLM safety by integrating representation engineering with model-based abstraction, paving the way for new paradigms to utilize software insights for AI safety. Our code is available at https://github.com/weizeming/ReGA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01734v1">Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive performance on complex reasoning tasks, yet they frequently fail on basic numerical problems, producing incorrect outputs. Inspired by Benford's Law -- a statistical pattern where lower digits occur more frequently as leading digits -- we hypothesize that the long-tailed digit distributions in web-collected corpora may be learned by LLMs during pretraining, leading to biased numerical generation. To investigate the hypothesis, we first examine whether digits frequencies in pretraining corpus (OLMo2) follows Benford's law. We then construct an evaluation benchmark with uniformly distributed ground-truth digits across seven numerical reasoning tasks. Our evaluation results demonstrate that leading open-source LLMs show a consistent pattern of digit bias that resembles Benford's law. Through logit-lens tracing and neuron-level dissection, we identify that this bias arises predominantly from a small subset of highly digit-selective feed-forward network (FFN) neurons in the deeper layers. Finally, we demonstrate that pruning these neurons mitigates imbalanced overgeneration and partially corrects erroneous outputs, providing causal evidence that fine-grained pretraining digit bias can propagate into model behavior. Our findings reveal a fundamental connection between corpus-level statistics and symbolic failure modes in LLMs, offering a new lens for diagnosing and mitigating hallucinations in numerical tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01732v1">Common Corpus: The Largest Collection of Ethical Data for LLM Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are pre-trained on large amounts of data from different sources and domains. These data most often contain trillions of tokens with large portions of copyrighted or proprietary content, which hinders the usage of such models under AI legislation. This raises the need for truly open pre-training data that is compliant with the data security regulations. In this paper, we introduce Common Corpus, the largest open dataset for language model pre-training. The data assembled in Common Corpus are either uncopyrighted or under permissible licenses and amount to about two trillion tokens. The dataset contains a wide variety of languages, ranging from the main European languages to low-resource ones rarely present in pre-training datasets; in addition, it includes a large portion of code data. The diversity of data sources in terms of covered domains and time periods opens up the paths for both research and entrepreneurial needs in diverse areas of knowledge. In this technical report, we present the detailed provenance of data assembling and the details of dataset filtering and curation. Being already used by such industry leaders as Anthropic and multiple LLM training projects, we believe that Common Corpus will become a critical infrastructure for open science research in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01713v1">SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown promising capabilities in reasoning tasks, yet still struggle with complex problems requiring explicit self-reflection and self-correction, especially compared to their unimodal text-based counterparts. Existing reflection methods are simplistic and struggle to generate meaningful and instructive feedback, as the reasoning ability and knowledge limits of pre-trained models are largely fixed during initial training. To overcome these challenges, we propose Multimodal Self-Reflection enhanced reasoning with Group Relative Policy Optimization (SRPO), a two-stage reflection-aware reinforcement learning (RL) framework explicitly designed to enhance multimodal LLM reasoning. In the first stage, we construct a high-quality, reflection-focused dataset under the guidance of an advanced MLLM, which generates reflections based on initial responses to help the policy model learn both reasoning and self-reflection. In the second stage, we introduce a novel reward mechanism within the GRPO framework that encourages concise and cognitively meaningful reflection while avoiding redundancy. Extensive experiments across multiple multimodal reasoning benchmarks, including MathVista, MathVision, MathVerse, and MMMU-Pro, using Qwen-2.5-VL-7B and Qwen-2.5-VL-32B demonstrate that SRPO significantly outperforms state-of-the-art models, achieving notable improvements in both reasoning accuracy and reflection quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01702v1">mdok of KInIT: Robustly Fine-tuned LLM for Binary and Multiclass AI-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The large language models (LLMs) are able to generate high-quality texts in multiple languages. Such texts are often not recognizable by humans as generated, and therefore present a potential of LLMs for misuse (e.g., plagiarism, spams, disinformation spreading). An automated detection is able to assist humans to indicate the machine-generated texts; however, its robustness to out-of-distribution data is still challenging. This notebook describes our mdok approach in robust detection, based on fine-tuning smaller LLMs for text classification. It is applied to both subtasks of Voight-Kampff Generative AI Detection 2025, providing remarkable performance in binary detection as well as in multiclass (1st rank) classification of various cases of human-AI collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01698v1">When LLMs Team Up: The Emergence of Collaborative Affective Computing</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 7 figures, and 3 tables
    </div>
    <details class="paper-abstract">
      Affective Computing (AC) is essential in bridging the gap between human emotional experiences and machine understanding. Traditionally, AC tasks in natural language processing (NLP) have been approached through pipeline architectures, which often suffer from structure rigidity that leads to inefficiencies and limited adaptability. The advent of Large Language Models (LLMs) has revolutionized this field by offering a unified approach to affective understanding and generation tasks, enhancing the potential for dynamic, real-time interactions. However, LLMs face cognitive limitations in affective reasoning, such as misinterpreting cultural nuances or contextual emotions, and hallucination problems in decision-making. To address these challenges, recent research advocates for LLM-based collaboration systems that emphasize interactions among specialized models and LLMs, mimicking human-like affective intelligence through the synergy of emotional and rational thinking that aligns with Dual Process Theory in psychology. This survey aims to provide a comprehensive overview of LLM-based collaboration systems in AC, exploring from structured collaborations to autonomous collaborations. Specifically, it includes: (1) A systematic review of existing methods, focusing on collaboration strategies, mechanisms, key functions, and applications; (2) Experimental comparisons of collaboration strategies across representative tasks in affective understanding and generation; (3) An analysis highlighting the potential of these systems to enhance robustness and adaptability in complex affective reasoning; (4) A discussion of key challenges and future research directions to further advance the field. This work is the first to systematically explore collaborative intelligence with LLMs in AC, paving the way for more powerful applications that approach human-like social intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02089v1">SALAD: Systematic Assessment of Machine Unlearing on LLM-Aided Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer transformative capabilities for hardware design automation, particularly in Verilog code generation. However, they also pose significant data security challenges, including Verilog evaluation data contamination, intellectual property (IP) design leakage, and the risk of malicious Verilog generation. We introduce SALAD, a comprehensive assessment that leverages machine unlearning to mitigate these threats. Our approach enables the selective removal of contaminated benchmarks, sensitive IP and design artifacts, or malicious code patterns from pre-trained LLMs, all without requiring full retraining. Through detailed case studies, we demonstrate how machine unlearning techniques effectively reduce data security risks in LLM-aided hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01687v1">StochasTok: Improving Fine-Grained Subword Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Subword-level understanding is integral to numerous tasks, including understanding multi-digit numbers, spelling mistakes, abbreviations, rhyming, and wordplay. Despite this, current large language models (LLMs) still often struggle with seemingly simple subword-level tasks like How many 'r's in 'strawberry'?. A key factor behind these failures is tokenization which obscures the fine-grained structure of words. Current alternatives, such as character-level and dropout tokenization methods, significantly increase computational costs and provide inconsistent improvements. In this paper we revisit tokenization and introduce StochasTok, a simple, efficient stochastic tokenization scheme that randomly splits tokens during training, allowing LLMs to 'see' their internal structure. Our experiments show that pretraining with StochasTok substantially improves LLMs' downstream performance across multiple subword-level language games, including character counting, substring identification, and math tasks. Furthermore, StochasTok's simplicity allows seamless integration at any stage of the training pipeline; and we demonstrate that post-training with StochasTok can instill improved subword understanding into existing pretrained models, thus avoiding costly pretraining from scratch. These dramatic improvements achieved with a minimal change suggest StochasTok holds exciting potential when applied to larger, more capable models. Code open-sourced at: https://github.com/anyasims/stochastok.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01646v1">ESGenius: Benchmarking LLMs on Environmental, Social, and Governance (ESG) and Sustainability Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 37 pages, 8 figures, 11 tables
    </div>
    <details class="paper-abstract">
      We introduce ESGenius, a comprehensive benchmark for evaluating and enhancing the proficiency of Large Language Models (LLMs) in Environmental, Social and Governance (ESG) and sustainability-focused question answering. ESGenius comprises two key components: (i) ESGenius-QA, a collection of 1 136 multiple-choice questions generated by LLMs and rigorously validated by domain experts, covering a broad range of ESG pillars and sustainability topics. Each question is systematically linked to its corresponding source text, enabling transparent evaluation and supporting retrieval-augmented generation (RAG) methods; and (ii) ESGenius-Corpus, a meticulously curated repository of 231 foundational frameworks, standards, reports and recommendation documents from seven authoritative sources. Moreover, to fully assess the capabilities and adaptation potential of the model, we implement a rigorous two-stage evaluation protocol -- Zero-Shot and RAG. Extensive experiments across 50 LLMs (ranging from 0.5 B to 671 B parameters) demonstrate that state-of-the-art models achieve only moderate performance in zero-shot settings, with accuracies typically around 55--70\%, highlighting ESGenius's challenging nature for LLMs in interdisciplinary contexts. However, models employing RAG show significant performance improvements, particularly for smaller models. For example, "DeepSeek-R1-Distill-Qwen-14B" improves from 63.82\% (zero-shot) to 80.46\% with RAG. These results underscore the necessity of grounding responses in authoritative sources for enhanced ESG understanding. To the best of our knowledge, ESGenius is the first benchmark curated for LLMs and the relevant enhancement technologies that focuses on ESG and sustainability topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01631v1">Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance. To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01616v1">MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      The emergence of multimodal LLM-based agents (MLAs) has transformed interaction paradigms by seamlessly integrating vision, language, action and dynamic environments, enabling unprecedented autonomous capabilities across GUI applications ranging from web automation to mobile systems. However, MLAs introduce critical trustworthiness challenges that extend far beyond traditional language models' limitations, as they can directly modify digital states and trigger irreversible real-world consequences. Existing benchmarks inadequately tackle these unique challenges posed by MLAs' actionable outputs, long-horizon uncertainty and multimodal attack vectors. In this paper, we introduce MLA-Trust, the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: truthfulness, controllability, safety and privacy. We utilize websites and mobile applications as realistic testbeds, designing 34 high-risk interactive tasks and curating rich evaluation datasets. Large-scale experiments involving 13 state-of-the-art agents reveal previously unexplored trustworthiness vulnerabilities unique to multimodal interactive scenarios. For instance, proprietary and open-source GUI-interacting MLAs pose more severe trustworthiness risks than static MLLMs, particularly in high-stakes domains; the transition from static MLLMs into interactive MLAs considerably compromises trustworthiness, enabling harmful content generation in multi-step interactions that standalone MLLMs would typically prevent; multi-step execution, while enhancing the adaptability of MLAs, involves latent nonlinear risk accumulation across successive interactions, circumventing existing safeguards and resulting in unpredictable derived risks. Moreover, we present an extensible toolbox to facilitate continuous evaluation of MLA trustworthiness across diverse interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v1">EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01484v1">LLM in the Loop: Creating the PARADEHATE Dataset for Hate Speech Detoxification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct PARADEHATE, a large-scale parallel dataset specifically for hatespeech detoxification. We release PARADEHATE as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on PARADEHATE, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01407v1">Comparing LLM-generated and human-authored news text using formal syntactic theory</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 15 figures, 13 tables; accepted to ACL-2025 main
    </div>
    <details class="paper-abstract">
      This study provides the first comprehensive comparison of New York Times-style text generated by six large language models against real, human-authored NYT writing. The comparison is based on a formal syntactic theory. We use Head-driven Phrase Structure Grammar (HPSG) to analyze the grammatical structure of the texts. We then investigate and illustrate the differences in the distributions of HPSG grammar types, revealing systematic distinctions between human and LLM-generated writing. These findings contribute to a deeper understanding of the syntactic behavior of LLMs as well as humans, within the NYT genre.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01386v1">ThinkEval: Practical Evaluation of Knowledge Preservation and Consistency in LLM Editing with Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Model editing has become an important tool for addressing privacy, bias, and misinformation in large language models (LLMs) by enabling updates to knowledge without the need for retraining from scratch. However, existing editing techniques often target isolated facts, ignoring ripple effects on related knowledge, allowing edited facts to remain deducible and compromising broader contextual integrity. For example, changing Harry Potter's school from Hogwarts to Ilvermorny requires reassigning his house from Gryffindor to a suitable alternative while preserving Gryffindor's relationship with Hogwarts. In this work, we present a new model-editing setting, deep editing, to show: (1) how editing techniques fail to handle connected facts, evaluating how original knowledge sneaks through unchanged causal links, and (2) their impact on broader contextual knowledge. We introduce ThinkEval, a framework to systematically evaluate model-editing techniques by building model-specific knowledge graphs to analyze pre- and post-edit effects on fact persistence and catastrophic forgetting. We present KnowGIC, a benchmark created with ThinkEval, consisting of sequentially linked queries to measure these effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. We find that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge. Our dataset is available at: https://anonymous.4open.science/r/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01374v1">Compiler Optimization via LLM Reasoning for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimization to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed REASONING COMPILER) that formulates optimization as a sequential, context-aware decision process, guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-aware transformations that reflect the current program state and accumulated performance feedback. Monte Carlo tree search (MCTS) incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01370v1">PointT2I: LLM-based text-to-image generation via keypoints</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generation model has made significant advancements, resulting in high-quality images aligned with an input prompt. However, despite T2I generation's ability to generate fine-grained images, it still faces challenges in accurately generating images when the input prompt contains complex concepts, especially human pose. In this paper, we propose PointT2I, a framework that effectively generates images that accurately correspond to the human pose described in the prompt by using a large language model (LLM). PointT2I consists of three components: Keypoint generation, Image generation, and Feedback system. The keypoint generation uses an LLM to directly generate keypoints corresponding to a human pose, solely based on the input prompt, without external references. Subsequently, the image generation produces images based on both the text prompt and the generated keypoints to accurately reflect the target pose. To refine the outputs of the preceding stages, we incorporate an LLM-based feedback system that assesses the semantic consistency between the generated contents and the given prompts. Our framework is the first approach to leveraging LLM for keypoints-guided image generation without any fine-tuning, producing accurate pose-aligned images based solely on textual prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01369v1">Incentivizing LLMs to Self-Verify Their Answers</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance during inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating its capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling. Our code is available at https://github.com/mansicer/self-verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01339v1">Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Machine unlearning offers a promising solution to privacy and safety concerns in large language models (LLMs) by selectively removing targeted knowledge while preserving utility. However, current methods are highly sensitive to downstream fine-tuning, which can quickly recover forgotten information-even from unrelated tasks. To address this, we introduce invariance into unlearning for the first time, inspired by invariant risk minimization (IRM). Building on this principle, we propose invariant LLM unlearning (ILU), a regularization-based framework that enhances robustness. Notably, ILU generalizes well to diverse fine-tuning tasks, even when trained using a single dataset. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP and MUSE benchmark, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01334v1">Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Concept Bottleneck Models (CBMs) decompose image classification into a process governed by interpretable, human-readable concepts. Recent advances in CBMs have used Large Language Models (LLMs) to generate candidate concepts. However, a critical question remains: What is the optimal number of concepts to use? Current concept banks suffer from redundancy or insufficient coverage. To address this issue, we introduce a dynamic, agent-based approach that adjusts the concept bank in response to environmental feedback, optimizing the number of concepts for sufficiency yet concise coverage. Moreover, we propose Conditional Concept Bottleneck Models (CoCoBMs) to overcome the limitations in traditional CBMs' concept scoring mechanisms. It enhances the accuracy of assessing each concept's contribution to classification tasks and feature an editable matrix that allows LLMs to correct concept scores that conflict with their internal knowledge. Our evaluations across 6 datasets show that our method not only improves classification accuracy by 6% but also enhances interpretability assessments by 30%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01290v1">TSRating: Rating Quality of Diverse Time Series Data by Meta-learning from LLM Judgment</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      High-quality time series (TS) data are essential for ensuring TS model performance, rendering research on rating TS data quality indispensable. Existing methods have shown promising rating accuracy within individual domains, primarily by extending data quality rating techniques such as influence functions and Shapley values to account for temporal characteristics. However, they neglect the fact that real-world TS data can span vastly different domains and exhibit distinct properties, hampering the accurate and efficient rating of diverse TS data. In this paper, we propose TSRating, a novel and unified framework for rating the quality of time series data crawled from diverse domains. TSRating is built on the assumption that LLMs inherit ample knowledge, acquired during their extensive pretraining, enabling them to comprehend and discern quality differences in diverse TS data. We verify this assumption by devising a series of prompts to elicit quality comparisons from LLMs for pairs of TS samples. We then fit a dedicated rating model, termed TSRater, to convert the LLMs' judgments into efficient quality predictions via TSRater's inference on future TS samples. To ensure cross-domain adaptability, we develop a meta-learning scheme to train TSRater on quality comparisons collected from nine distinct domains. To improve training efficiency, we employ signSGD for inner-loop updates, thus circumventing the demanding computation of hypergradients. Extensive experimental results on eleven benchmark datasets across three time series tasks, each using both conventional TS models and TS foundation models, demonstrate that TSRating outperforms baselines in terms of estimation accuracy, efficiency, and domain adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01262v1">Exploring the Potential of LLMs as Personalized Assistants: Dataset, Evaluation, and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Personalized AI assistants, a hallmark of the human-like capabilities of Large Language Models (LLMs), are a challenging application that intertwines multiple problems in LLM research. Despite the growing interest in the development of personalized assistants, the lack of an open-source conversational dataset tailored for personalization remains a significant obstacle for researchers in the field. To address this research gap, we introduce HiCUPID, a new benchmark to probe and unleash the potential of LLMs to deliver personalized responses. Alongside a conversational dataset, HiCUPID provides a Llama-3.2-based automated evaluation model whose assessment closely mirrors human preferences. We release our dataset, evaluation model, and code at https://github.com/12kimih/HiCUPID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01252v1">MTCMB: A Multi-Task Benchmark Framework for Evaluating LLMs on Knowledge, Reasoning, and Safety in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) is a holistic medical system with millennia of accumulated clinical experience, playing a vital role in global healthcare-particularly across East Asia. However, the implicit reasoning, diverse textual forms, and lack of standardization in TCM pose major challenges for computational modeling and evaluation. Large Language Models (LLMs) have demonstrated remarkable potential in processing natural language across diverse domains, including general medicine. Yet, their systematic evaluation in the TCM domain remains underdeveloped. Existing benchmarks either focus narrowly on factual question answering or lack domain-specific tasks and clinical realism. To fill this gap, we introduce MTCMB-a Multi-Task Benchmark for Evaluating LLMs on TCM Knowledge, Reasoning, and Safety. Developed in collaboration with certified TCM experts, MTCMB comprises 12 sub-datasets spanning five major categories: knowledge QA, language understanding, diagnostic reasoning, prescription generation, and safety evaluation. The benchmark integrates real-world case records, national licensing exams, and classical texts, providing an authentic and comprehensive testbed for TCM-capable models. Preliminary results indicate that current LLMs perform well on foundational knowledge but fall short in clinical reasoning, prescription planning, and safety compliance. These findings highlight the urgent need for domain-aligned benchmarks like MTCMB to guide the development of more competent and trustworthy medical AI systems. All datasets, code, and evaluation tools are publicly available at: https://github.com/Wayyuanyuan/MTCMB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01245v1">Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01237v1">Polishing Every Facet of the GEM: Testing Linguistic Competence of LLMs and Humans in Korean</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted at ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      We introduce the $\underline{Ko}rean \underline{G}rammar \underline{E}valuation Bench\underline{M}ark (KoGEM)$, designed to assess the linguistic competence of LLMs and humans in Korean. KoGEM consists of 1.5k multiple-choice QA pairs covering five main categories and 16 subcategories. The zero-shot evaluation of 27 LLMs of various sizes and types reveals that while LLMs perform remarkably well on straightforward tasks requiring primarily definitional knowledge, they struggle with tasks that demand the integration of real-world experiential knowledge, such as phonological rules and pronunciation. Furthermore, our in-depth analysis suggests that incorporating such experiential knowledge could enhance the linguistic competence of LLMs. With KoGEM, we not only highlight the limitations of current LLMs in linguistic competence but also uncover hidden facets of LLMs in linguistic competence, paving the way for enhancing comprehensive language understanding. Our code and dataset are available at: https://github.com/SungHo3268/KoGEM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23799v2">Estimating LLM Consistency: A User Baseline vs Surrogate Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are prone to hallucinations and sensitive to prompt perturbations, often resulting in inconsistent or unreliable generated text. Different methods have been proposed to mitigate such hallucinations and fragility -- one of them being measuring the consistency (the model's confidence in the response, or likelihood of generating a similar response when resampled) of LLM responses. In previous work, measuring consistency often relied on the probability of a response appearing within a pool of resampled responses, or internal states or logits of responses. However, it is not yet clear how well these approaches approximate how humans perceive the consistency of LLM responses. We performed a user study (n=2,976) and found current methods typically do not approximate users' perceptions of LLM consistency very well. We propose a logit-based ensemble method for estimating LLM consistency, and we show that this method matches the performance of the best-performing existing metric in estimating human ratings of LLM consistency. Our results suggest that methods of estimating LLM consistency without human evaluation are sufficiently imperfect that we suggest evaluation with human input be more broadly used.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07071v3">Value Compass Benchmarks: A Platform for Fundamental and Validated Evaluation of LLMs Values</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) achieve remarkable breakthroughs, aligning their values with humans has become imperative for their responsible development and customized applications. However, there still lack evaluations of LLMs values that fulfill three desirable goals. (1) Value Clarification: We expect to clarify the underlying values of LLMs precisely and comprehensively, while current evaluations focus narrowly on safety risks such as bias and toxicity. (2) Evaluation Validity: Existing static, open-source benchmarks are prone to data contamination and quickly become obsolete as LLMs evolve. Additionally, these discriminative evaluations uncover LLMs' knowledge about values, rather than valid assessments of LLMs' behavioral conformity to values. (3) Value Pluralism: The pluralistic nature of human values across individuals and cultures is largely ignored in measuring LLMs value alignment. To address these challenges, we presents the Value Compass Benchmarks, with three correspondingly designed modules. It (i) grounds the evaluation on motivationally distinct \textit{basic values to clarify LLMs' underlying values from a holistic view; (ii) applies a \textit{generative evolving evaluation framework with adaptive test items for evolving LLMs and direct value recognition from behaviors in realistic scenarios; (iii) propose a metric that quantifies LLMs alignment with a specific value as a weighted sum over multiple dimensions, with weights determined by pluralistic values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07453v2">How well do LLMs reason over tabular data, really?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07784v2">Domain Regeneration: How well do LLMs match syntactic properties of text domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Recent improvement in large language model performance have, in all likelihood, been accompanied by improvement in how well they can approximate the distribution of their training data. In this work, we explore the following question: which properties of text domains do LLMs faithfully approximate, and how well do they do so? Applying observational approaches familiar from corpus linguistics, we prompt a commonly used, opensource LLM to regenerate text from two domains of permissively licensed English text which are often contained in LLM training data -- Wikipedia and news text. This regeneration paradigm allows us to investigate whether LLMs can faithfully match the original human text domains in a fairly semantically-controlled setting. We investigate varying levels of syntactic abstraction, from more simple properties like sentence length, and article readability, to more complex and higher order properties such as dependency tag distribution, parse depth, and parse complexity. We find that the majority of the regenerated distributions show a shifted mean, a lower standard deviation, and a reduction of the long tail, as compared to the human originals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04503v3">When LLMs Play the Telephone Game: Cultural Attractors as Conceptual Tools to Evaluate LLMs in Multi-turn Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Code available at https://github.com/jeremyperez2/TelephoneGameLLM. Companion website with a Data Explorer tool at https://sites.google.com/view/telephone-game-llm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) start interacting with each other and generating an increasing amount of text online, it becomes crucial to better understand how information is transformed as it passes from one LLM to the next. While significant research has examined individual LLM behaviors, existing studies have largely overlooked the collective behaviors and information distortions arising from iterated LLM interactions. Small biases, negligible at the single output level, risk being amplified in iterated interactions, potentially leading the content to evolve towards attractor states. In a series of telephone game experiments, we apply a transmission chain design borrowed from the human cultural evolution literature: LLM agents iteratively receive, produce, and transmit texts from the previous to the next agent in the chain. By tracking the evolution of text toxicity, positivity, difficulty, and length across transmission chains, we uncover the existence of biases and attractors, and study their dependence on the initial text, the instructions, language model, and model size. For instance, we find that more open-ended instructions lead to stronger attraction effects compared to more constrained tasks. We also find that different text properties display different sensitivity to attraction effects, with toxicity leading to stronger attractors than length. These findings highlight the importance of accounting for multi-step transmission dynamics and represent a first step towards a more comprehensive understanding of LLM cultural dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03340v2">EnigmaToM: Improve LLMs' Theory-of-Mind Reasoning Capabilities with Neural Knowledge Base of Entity States</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Theory-of-Mind (ToM), the ability to infer others' perceptions and mental states, is fundamental to human interaction but remains challenging for Large Language Models (LLMs). While existing ToM reasoning methods show promise with reasoning via perceptual perspective-taking, they often rely excessively on off-the-shelf LLMs, reducing their efficiency and limiting their applicability to high-order ToM reasoning. To address these issues, we present EnigmaToM, a novel neuro-symbolic framework that enhances ToM reasoning by integrating a Neural Knowledge Base of entity states (Enigma) for (1) a psychology-inspired iterative masking mechanism that facilitates accurate perspective-taking and (2) knowledge injection that elicits key entity information. Enigma generates structured knowledge of entity states to build spatial scene graphs for belief tracking across various ToM orders and enrich events with fine-grained entity state details. Experimental results on ToMi, HiToM, and FANToM benchmarks show that EnigmaToM significantly improves ToM reasoning across LLMs of varying sizes, particularly excelling in high-order reasoning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09439v2">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU and MMAR benchmarks. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01150v2">MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025 (Findings) Code and data available at https://github.com/luciusssss/MiLiC-Eval
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in high-resource languages but struggle with low-resource languages (LRLs), particularly those spoken by minority communities in China, such as Tibetan, Uyghur, Kazakh, and Mongolian. To systematically track the progress in these languages, we introduce MiLiC-Eval, a benchmark designed for minority languages in China, featuring 24K instances across 9 tasks. MiLiC-Eval focuses on underrepresented writing systems. Its parallelism between tasks and languages can provide a faithful and fine-grained assessment of linguistic and problem-solving skills. Our evaluation reveals that open-source LLMs perform poorly on syntax-intensive tasks and multi-script languages. We further demonstrate how MiLiC-Eval can help advance LRL research in handling diverse writing systems and understanding the process of language adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15268v4">Enhancing LLM-based Hatred and Toxicity Detection with Meta-Toxic Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 8 pages of content
    </div>
    <details class="paper-abstract">
      The rapid growth of social media platforms has raised significant concerns regarding online content toxicity. When Large Language Models (LLMs) are used for toxicity detection, two key challenges emerge: 1) the absence of domain-specific toxic knowledge leads to false negatives; 2) the excessive sensitivity of LLMs to toxic speech results in false positives, limiting freedom of speech. To address these issues, we propose a novel method called MetaTox, leveraging graph search on a meta-toxic knowledge graph to enhance hatred and toxicity detection. First, we construct a comprehensive meta-toxic knowledge graph by utilizing LLMs to extract toxic information through a three-step pipeline, with toxic benchmark datasets serving as corpora. Second, we query the graph via retrieval and ranking processes to supplement accurate, relevant toxic knowledge. Extensive experiments and in-depth case studies across multiple datasets demonstrate that our MetaTox significantly decreases the false positive rate while boosting overall toxicity detection performance. Our code is available at https://github.com/YiboZhao624/MetaTox.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18403v3">LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to the main conference of ACL 2025
    </div>
    <details class="paper-abstract">
      There is an increasing trend towards evaluating NLP models with LLMs instead of human judgments, raising questions about the validity of these evaluations, as well as their reproducibility in the case of proprietary models. We provide JUDGE-BENCH, an extensible collection of 20 NLP datasets with human annotations covering a broad range of evaluated properties and types of data, and comprehensively evaluate 11 current LLMs, covering both open-weight and proprietary models, for their ability to replicate the annotations. Our evaluations show substantial variance across models and datasets. Models are reliable evaluators on some tasks, but overall display substantial variability depending on the property being evaluated, the expertise level of the human judges, and whether the language is human or model-generated. We conclude that LLMs should be carefully validated against human judgments before being used as evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14050v4">Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted to the Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. We reduce biases by finetuning on curated non-harmful text, but find only direct preference optimization to be effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07217v3">ReelWave: Multi-Agentic Movie Sound Generation through Multimodal LLM Conversation</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Project page: https://vincent2311.github.io/ReelWave_demo
    </div>
    <details class="paper-abstract">
      Current audio generation conditioned by text or video focuses on aligning audio with text/video modalities. Despite excellent alignment results, these multimodal frameworks still cannot be directly applied to compelling movie storytelling involving multiple scenes, where "on-screen" sounds require temporally-aligned audio generation, while "off-screen" sounds contribute to appropriate environment sounds accompanied by background music when applicable. Inspired by professional movie production, this paper proposes a multi-agentic framework for audio generation supervised by an autonomous Sound Director agent, engaging multi-turn conversations with other agents for on-screen and off-screen sound generation through multimodal LLM. To address on-screen sound generation, after detecting any talking humans in videos, we capture semantically and temporally synchronized sound by training a prediction model that forecasts interpretable, time-varying audio control signals: loudness, pitch, and timbre, which are used by a Foley Artist agent to condition a cross-attention module in the sound generation. The Foley Artist works cooperatively with the Composer and Voice Actor agents, and together they autonomously generate off-screen sound to complement the overall production. Each agent takes on specific roles similar to those of a movie production team. To temporally ground audio language models, in ReelWave, text/video conditions are decomposed into atomic, specific sound generation instructions synchronized with visuals when applicable. Consequently, our framework can generate rich and relevant audio content conditioned on video clips extracted from movies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24362v2">Knowing Before Saying: LLM Representations Encode Information About Chain-of-Thought Success Before Completion</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      We investigate whether the success of a zero-shot Chain-of-Thought (CoT) process can be predicted before completion. We discover that a probing classifier, based on LLM representations, performs well \emph{even before a single token is generated}, suggesting that crucial information about the reasoning process is already present in the initial steps representations. In contrast, a strong BERT-based baseline, which relies solely on the generated tokens, performs worse, likely because it depends on shallow linguistic cues rather than deeper reasoning dynamics. Surprisingly, using later reasoning steps does not always improve classification. When additional context is unhelpful, earlier representations resemble later ones more, suggesting LLMs encode key information early. This implies reasoning can often stop early without loss. To test this, we conduct early stopping experiments, showing that truncating CoT reasoning still improves performance over not using CoT at all, though a gap remains compared to full reasoning. However, approaches like supervised learning or reinforcement learning designed to shorten CoT chains could leverage our classifier's guidance to identify when early stopping is effective. Our findings provide insights that may support such methods, helping to optimize CoT's efficiency while preserving its benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15865v2">Standard Benchmarks Fail -- Auditing LLM Agents in Finance Must Prioritize Risk</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 46 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Standard benchmarks fixate on how well large language model (LLM) agents perform in finance, yet say little about whether they are safe to deploy. We argue that accuracy metrics and return-based scores provide an illusion of reliability, overlooking vulnerabilities such as hallucinated facts, stale data, and adversarial prompt manipulation. We take a firm position: financial LLM agents should be evaluated first and foremost on their risk profile, not on their point-estimate performance. Drawing on risk-engineering principles, we outline a three-level agenda: model, workflow, and system, for stress-testing LLM agents under realistic failure modes. To illustrate why this shift is urgent, we audit six API-based and open-weights LLM agents on three high-impact tasks and uncover hidden weaknesses that conventional benchmarks miss. We conclude with actionable recommendations for researchers, practitioners, and regulators: audit risk-aware metrics in future studies, publish stress scenarios alongside datasets, and treat ``safety budget'' as a primary success criterion. Only by redefining what ``good'' looks like can the community responsibly advance AI-driven finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16310v4">An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 21 pages, 5 images, 8 tables, Manuscript submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Security code review is a time-consuming and labor-intensive process typically requiring integration with automated security defect detection tools. However, existing security analysis tools struggle with poor generalization, high false positive rates, and coarse detection granularity. Large Language Models (LLMs) have been considered promising candidates for addressing those challenges. In this study, we conducted an empirical study to explore the potential of LLMs in detecting security defects during code review. Specifically, we evaluated the performance of six LLMs under five different prompts and compared them with state-of-the-art static analysis tools. We also performed linguistic and regression analyses for the best-performing LLM to identify quality problems in its responses and factors influencing its performance. Our findings showthat: (1) existing pre-trained LLMs have limited capability in security code review but significantly outperformthe state-of-the-art static analysis tools. (2) GPT-4 performs best among all LLMs when provided with a CWE list for reference. (3) GPT-4 frequently generates verbose or non-compliant responses with the task requirements given in the prompts. (4) GPT-4 is more adept at identifying security defects in code files with fewer tokens, containing functional logic, or written by developers with less involvement in the project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13899v2">Schemato -- An LLM for Netlist-to-Schematic Conversion</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Machine learning models are advancing circuit design, particularly in analog circuits. They typically generate netlists that lack human interpretability. This is a problem as human designers heavily rely on the interpretability of circuit diagrams or schematics to intuitively understand, troubleshoot, and develop designs. Hence, to integrate domain knowledge effectively, it is crucial to translate ML-generated netlists into interpretable schematics quickly and accurately. We propose Schemato, a large language model (LLM) for netlist-to-schematic conversion. In particular, we consider our approach in converting netlists to .asc files, text-based schematic description used in LTSpice. Experiments on our circuit dataset show that Schemato achieves up to 76% compilation success rate, surpassing 63% scored by the state-of-the-art LLMs. Furthermore, our experiments show that Schemato generates schematics with an average graph edit distance score and mean structural similarity index measure, scaled by the compilation success rate that are 1.8x and 4.3x higher than the best performing LLMs respectively, demonstrating its ability to generate schematics that are more accurately connected and are closer to the reference human design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08219v2">Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks, leading to widespread adoption in both research and industry. However, their inference workloads are computationally and energy intensive, raising concerns about sustainability and environmental impact. As LLMs continue to scale, it becomes essential to identify and optimize the factors that influence their runtime efficiency without compromising performance. In this work, we systematically investigate the energy-performance trade-offs of LLMs during inference. We benchmark models of varying sizes and architectures, including Falcon-7B, Mistral-7B-v0.1, LLaMA-3.2-1B, LLaMA-3.2-3B, and GPT-Neo-2.7B, across tasks such as question answering, commonsense reasoning, and factual generation. We analyze the effect of input characteristics, such as sequence length, entropy, named entity density and so on. Furthermore, we examine the impact of hardware-level optimizations through Dynamic Voltage and Frequency Scaling (DVFS), measuring how different GPU clock settings affect latency and power consumption. Our empirical findings show that model architecture, input complexity, and clock configuration significantly influence inference efficiency. By correlating input features with energy metrics and evaluating DVFS behavior, we identify practical strategies that reduce energy consumption by up to 30% while preserving model quality. This study provides actionable insights for designing energy-efficient and sustainable LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14830v3">Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (https://github.com/dannigt/mid-align).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00113v3">Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Accepted for ACL 2025 Main Conference and NeurIPS 2024 FM-EduAssess Workshop
    </div>
    <details class="paper-abstract">
      This work introduces a novel framework for evaluating LLMs' capacity to balance instruction-following with critical reasoning when presented with multiple-choice questions containing no valid answers. Through systematic evaluation across arithmetic, domain-specific knowledge, and high-stakes medical decision tasks, we demonstrate that post-training aligned models often default to selecting invalid options, while base models exhibit improved refusal capabilities that scale with model size. Our analysis reveals that alignment techniques, though intended to enhance helpfulness, can inadvertently impair models' reflective judgment--the ability to override default behaviors when faced with invalid options. We additionally conduct a parallel human study showing similar instruction-following biases, with implications for how these biases may propagate through human feedback datasets used in alignment. We provide extensive ablation studies examining the impact of model size, training techniques, and prompt engineering. Our findings highlight fundamental tensions between alignment optimization and preservation of critical reasoning capabilities, with important implications for developing more robust AI systems for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09923v2">Guiding Reasoning in Small Language Models with LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 20 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The limited reasoning capabilities of small language models (SLMs) cast doubt on their suitability for tasks demanding deep, multi-step logical deduction. This paper introduces a framework called Small Reasons, Large Hints (SMART), which selectively augments SLM reasoning with targeted guidance from large language models (LLMs). Inspired by the concept of cognitive scaffolding, SMART employs a score-based evaluation to identify uncertain reasoning steps and injects corrective LLM-generated reasoning only when necessary. By framing structured reasoning as an optimal policy search, our approach steers the reasoning trajectory toward correct solutions without exhaustive sampling. Our experiments on mathematical reasoning datasets demonstrate that targeted external scaffolding significantly improves performance, paving the way for collaborative use of both SLM and LLM to tackle complex reasoning tasks that are currently unsolvable by SLMs alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10201v2">Prediction hubs are context-informed frequent tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 Published as a conference paper at ACL 2025
    </div>
    <details class="paper-abstract">
      Hubness, the tendency for a few points to be among the nearest neighbours of a disproportionate number of other points, commonly arises when applying standard distance measures to high-dimensional data, often negatively impacting distance-based analysis. As autoregressive large language models (LLMs) operate on high-dimensional representations, we ask whether they are also affected by hubness. We first prove that the only large-scale representation comparison operation performed by LLMs, namely that between context and unembedding vectors to determine continuation probabilities, is not characterized by the concentration of distances phenomenon that typically causes the appearance of nuisance hubness. We then empirically show that this comparison still leads to a high degree of hubness, but the hubs in this case do not constitute a disturbance. They are rather the result of context-modulated frequent tokens often appearing in the pool of likely candidates for next token prediction. However, when other distances are used to compare LLM representations, we do not have the same theoretical guarantees, and, indeed, we see nuisance hubs appear. There are two main takeaways. First, hubness, while omnipresent in high-dimensional spaces, is not a negative property that needs to be mitigated when LLMs are being used for next token prediction. Second, when comparing representations from LLMs using Euclidean or cosine distance, there is a high risk of nuisance hubs and practitioners should use mitigation techniques if relevant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02508v2">Satori: Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capabilities across diverse domains. Recent studies have shown that increasing test-time computation enhances LLMs' reasoning capabilities. This typically involves extensive sampling at inference time guided by an external LLM verifier, resulting in a two-player system. Despite external guidance, the effectiveness of this system demonstrates the potential of a single LLM to tackle complex tasks. Thus, we pose a new research problem: Can we internalize the searching capabilities to fundamentally enhance the reasoning abilities of a single LLM? This work explores an orthogonal direction focusing on post-training LLMs for autoregressive searching (i.e., an extended reasoning process with self-reflection and self-exploration of new strategies). To achieve this, we propose the Chain-of-Action-Thought (COAT) reasoning and a two-stage training paradigm: 1) a small-scale format tuning stage to internalize the COAT reasoning format and 2) a large-scale self-improvement stage leveraging reinforcement learning. Our approach results in Satori, a 7B LLM trained on open-source models and data. Extensive empirical evaluations demonstrate that Satori achieves state-of-the-art performance on mathematical reasoning benchmarks while exhibits strong generalization to out-of-domain tasks. Code, data, and models are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00212v3">Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-02
      | 💬 camera-ready
    </div>
    <details class="paper-abstract">
      Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at https://github.com/mingyin1/Agents_Failure_Attribution
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18970v2">Learning to Explain: Prototype-Based Surrogate Models for LLM Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on natural language tasks, but their decision-making processes remain largely opaque. Existing explanation methods either suffer from limited faithfulness to the model's reasoning or produce explanations that humans find difficult to understand. To address these challenges, we propose \textbf{ProtoSurE}, a novel prototype-based surrogate framework that provides faithful and human-understandable explanations for LLMs. ProtoSurE trains an interpretable-by-design surrogate model that aligns with the target LLM while utilizing sentence-level prototypes as human-understandable concepts. Extensive experiments show that ProtoSurE consistently outperforms SOTA explanation methods across diverse LLMs and datasets. Importantly, ProtoSurE demonstrates strong data efficiency, requiring relatively few training examples to achieve good performance, making it practical for real-world applications.
    </details>
</div>
