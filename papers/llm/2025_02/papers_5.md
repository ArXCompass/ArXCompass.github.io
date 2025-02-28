# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14008v1">MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges for deployment and inference. Structured pruning, an effective model compression technique, is gaining increasing attention due to its ability to enhance inference efficiency. Nevertheless, most previous optimization-based structured pruning methods sacrifice the uniform structure across layers for greater flexibility to maintain performance. The heterogeneous structure hinders the effective utilization of off-the-shelf inference acceleration techniques and impedes efficient configuration for continued training. To address this issue, we propose a novel masking learning paradigm based on minimax optimization to obtain the uniform pruned structure by optimizing the masks under sparsity regularization. Extensive experimental results demonstrate that our method can maintain high performance while ensuring the uniformity of the pruned model structure, thereby outperforming existing SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13996v1">Beyond Single-Value Metrics: Evaluating and Enhancing LLM Unlearning with Cognitive Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Due to the widespread use of LLMs and the rising critical ethical and safety concerns, LLM unlearning methods have been developed to remove harmful knowledge and undesirable capabilities. In this context, evaluations are mostly based on single-value metrics such as QA accuracy. However, these metrics often fail to capture the nuanced retention of harmful knowledge components, making it difficult to assess the true effectiveness of unlearning. To address this issue, we propose UNCD (UNlearning evaluation via Cognitive Diagnosis), a novel framework that leverages Cognitive Diagnosis Modeling for fine-grained evaluation of LLM unlearning. Our dedicated benchmark, UNCD-Cyber, provides a detailed assessment of the removal of dangerous capabilities. Moreover, we introduce UNCD-Agent, which refines unlearning by diagnosing knowledge remnants and generating targeted unlearning data. Extensive experiments across eight unlearning methods and two base models demonstrate that UNCD not only enhances evaluation but also effectively facilitates the removal of harmful LLM abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13965v1">Autellix: An Efficient Serving Engine for LLM Agents as General Programs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) applications are evolving beyond simple chatbots into dynamic, general-purpose agentic programs, which scale LLM calls and output tokens to help AI agents reason, explore, and solve complex tasks. However, existing LLM serving systems ignore dependencies between programs and calls, missing significant opportunities for optimization. Our analysis reveals that programs submitted to LLM serving engines experience long cumulative wait times, primarily due to head-of-line blocking at both the individual LLM request and the program. To address this, we introduce Autellix, an LLM serving system that treats programs as first-class citizens to minimize their end-to-end latencies. Autellix intercepts LLM calls submitted by programs, enriching schedulers with program-level context. We propose two scheduling algorithms-for single-threaded and distributed programs-that preempt and prioritize LLM calls based on their programs' previously completed calls. Our evaluation demonstrates that across diverse LLMs and agentic workloads, Autellix improves throughput of programs by 4-15x at the same latency compared to state-of-the-art systems, such as vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02890v4">Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. In this paper, we present a novel theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and the detection process. Our approach focuses on maximizing detection performance while maintaining control over the worst-case Type-I error and text distortion. We characterize \emph{the universally minimum Type-II error}, showing a fundamental trade-off between watermark detectability and text distortion. Importantly, we identify that the optimal watermarking schemes are adaptive to the LLM generative distribution. Building on our theoretical insights, we propose an efficient, model-agnostic, distribution-adaptive watermarking algorithm, utilizing a surrogate model alongside the Gumbel-max trick. Experiments conducted on Llama2-13B and Mistral-8$\times$7B models confirm the effectiveness of our approach. Additionally, we examine incorporating robustness into our framework, paving a way to future watermarking systems that withstand adversarial attacks more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13913v1">How Do LLMs Perform Two-Hop Reasoning in Context?</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      "Socrates is human. All humans are mortal. Therefore, Socrates is mortal." This classical example demonstrates two-hop reasoning, where a conclusion logically follows from two connected premises. While transformer-based Large Language Models (LLMs) can make two-hop reasoning, they tend to collapse to random guessing when faced with distracting premises. To understand the underlying mechanism, we train a three-layer transformer on synthetic two-hop reasoning tasks. The training dynamics show two stages: a slow learning phase, where the 3-layer transformer performs random guessing like LLMs, followed by an abrupt phase transitions, where the 3-layer transformer suddenly reaches $100%$ accuracy. Through reverse engineering, we explain the inner mechanisms for how models learn to randomly guess between distractions initially, and how they learn to ignore distractions eventually. We further propose a three-parameter model that supports the causal claims for the mechanisms to the training dynamics of the transformer. Finally, experiments on LLMs suggest that the discovered mechanisms generalize across scales. Our methodologies provide new perspectives for scientific understandings of LLMs and our findings provide new insights into how reasoning emerges during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13908v1">Judging the Judges: A Collection of LLM-Generated Relevance Judgements</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) for relevance assessments offers promising opportunities to improve Information Retrieval (IR), Natural Language Processing (NLP), and related fields. Indeed, LLMs hold the promise of allowing IR experimenters to build evaluation collections with a fraction of the manual human labor currently required. This could help with fresh topics on which there is still limited knowledge and could mitigate the challenges of evaluating ranking systems in low-resource scenarios, where it is challenging to find human annotators. Given the fast-paced recent developments in the domain, many questions concerning LLMs as assessors are yet to be answered. Among the aspects that require further investigation, we can list the impact of various components in a relevance judgment generation pipeline, such as the prompt used or the LLM chosen. This paper benchmarks and reports on the results of a large-scale automatic relevance judgment evaluation, the LLMJudge challenge at SIGIR 2024, where different relevance assessment approaches were proposed. In detail, we release and benchmark 42 LLM-generated labels of the TREC 2023 Deep Learning track relevance judgments produced by eight international teams who participated in the challenge. Given their diverse nature, these automatically generated relevance judgments can help the community not only investigate systematic biases caused by LLMs but also explore the effectiveness of ensemble models, analyze the trade-offs between different models and human assessors, and advance methodologies for improving automated evaluation techniques. The released resource is available at the following link: https://llm4eval.github.io/LLMJudge-benchmark/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13897v1">DataSciBench: An LLM Agent Benchmark for Data Science</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 40 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This paper presents DataSciBench, a comprehensive benchmark for evaluating Large Language Model (LLM) capabilities in data science. Recent related benchmarks have primarily focused on single tasks, easily obtainable ground truth, and straightforward evaluation metrics, which limits the scope of tasks that can be evaluated. In contrast, DataSciBench is constructed based on a more comprehensive and curated collection of natural and challenging prompts for uncertain ground truth and evaluation metrics. We develop a semi-automated pipeline for generating ground truth (GT) and validating evaluation metrics. This pipeline utilizes and implements an LLM-based self-consistency and human verification strategy to produce accurate GT by leveraging collected prompts, predefined task types, and aggregate functions (metrics). Furthermore, we propose an innovative Task - Function - Code (TFC) framework to assess each code execution outcome based on precisely defined metrics and programmatic rules. Our experimental framework involves testing 6 API-based models, 8 open-source general models, and 9 open-source code generation models using the diverse set of prompts we have gathered. This approach aims to provide a more comprehensive and rigorous evaluation of LLMs in data science, revealing their strengths and weaknesses. Experimental results demonstrate that API-based models outperform open-sourced models on all metrics and Deepseek-Coder-33B-Instruct achieves the highest score among open-sourced models. We release all code and data at https://github.com/THUDM/DataSciBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13870v1">SPEX: Scaling Feature Interaction Explanations for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized machine learning due to their ability to capture complex interactions between input features. Popular post-hoc explanation methods like SHAP provide marginal feature attributions, while their extensions to interaction importances only scale to small input lengths ($\approx 20$). We propose Spectral Explainer (SPEX), a model-agnostic interaction attribution algorithm that efficiently scales to large input lengths ($\approx 1000)$. SPEX exploits underlying natural sparsity among interactions -- common in real-world data -- and applies a sparse Fourier transform using a channel decoding algorithm to efficiently identify important interactions. We perform experiments across three difficult long-context datasets that require LLMs to utilize interactions between inputs to complete the task. For large inputs, SPEX outperforms marginal attribution methods by up to 20% in terms of faithfully reconstructing LLM outputs. Further, SPEX successfully identifies key features and interactions that strongly influence model output. For one of our datasets, HotpotQA, SPEX provides interactions that align with human annotations. Finally, we use our model-agnostic approach to generate explanations to demonstrate abstract reasoning in closed-source LLMs (GPT-4o mini) and compositional reasoning in vision-language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13845v1">Enhancing LLM-Based Recommendations Through Personalized Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 7 pages, under review
    </div>
    <details class="paper-abstract">
      Current recommendation systems powered by large language models (LLMs) often underutilize their reasoning capabilities due to a lack of explicit logical structuring. To address this limitation, we introduce CoT-Rec, a framework that integrates Chain-of-Thought (CoT) reasoning into LLM-driven recommendations by incorporating two crucial processes: user preference analysis and item perception evaluation. CoT-Rec operates in two key phases: (1) personalized data extraction, where user preferences and item perceptions are identified, and (2) personalized data application, where this information is leveraged to refine recommendations. Our experimental analysis demonstrates that CoT-Rec improves recommendation accuracy by making better use of LLMs' reasoning potential. The implementation is publicly available at https://anonymous.4open.science/r/CoT-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13843v1">Enhancing Cross-Domain Recommendations with Memory-Optimized LLM-Based User Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 6 pages, under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based user agents have emerged as a powerful tool for improving recommender systems by simulating user interactions. However, existing methods struggle with cross-domain scenarios due to inefficient memory structures, leading to irrelevant information retention and failure to account for social influence factors such as popularity. To address these limitations, we introduce AgentCF++, a novel framework featuring a dual-layer memory architecture and a two-step fusion mechanism to filter domain-specific preferences effectively. Additionally, we propose interest groups with shared memory, allowing the model to capture the impact of popularity trends on users with similar interests. Through extensive experiments on multiple cross-domain datasets, AgentCF++ demonstrates superior performance over baseline models, highlighting its effectiveness in refining user behavior simulation for recommender systems. Our code is available at https://anonymous.4open.science/r/AgentCF-plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13834v1">Proving Olympiad Inequalities by Synergizing LLMs and Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Published as a conference paper at ICLR 2025. Code is available at https://github.com/Lizn-zn/NeqLIPS/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can prove mathematical theorems formally by generating proof steps (\textit{a.k.a.} tactics) within a proof system. However, the space of possible tactics is vast and complex, while the available training data for formal proofs is limited, posing a significant challenge to LLM-based tactic generation. To address this, we introduce a neuro-symbolic tactic generator that synergizes the mathematical intuition learned by LLMs with domain-specific insights encoded by symbolic methods. The key aspect of this integration is identifying which parts of mathematical reasoning are best suited to LLMs and which to symbolic methods. While the high-level idea of neuro-symbolic integration is broadly applicable to various mathematical problems, in this paper, we focus specifically on Olympiad inequalities (Figure~1). We analyze how humans solve these problems and distill the techniques into two types of tactics: (1) scaling, handled by symbolic methods, and (2) rewriting, handled by LLMs. In addition, we combine symbolic tools with LLMs to prune and rank the proof goals for efficient proof search. We evaluate our framework on 161 challenging inequalities from multiple mathematics competitions, achieving state-of-the-art performance and significantly outperforming existing LLM and symbolic approaches without requiring additional training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13794v1">LESA: Learnable LLM Layer Scaling-Up</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) from scratch requires immense computational resources, making it prohibitively expensive. Model scaling-up offers a promising solution by leveraging the parameters of smaller models to create larger ones. However, existing depth scaling-up methods rely on empirical heuristic rules for layer duplication, which result in poorer initialization and slower convergence during continual pre-training. We propose \textbf{LESA}, a novel learnable method for depth scaling-up. By concatenating parameters from each layer and applying Singular Value Decomposition, we uncover latent patterns between layers, suggesting that inter-layer parameters can be learned. LESA uses a neural network to predict the parameters inserted between adjacent layers, enabling better initialization and faster training. Experiments show that LESA outperforms existing baselines, achieving superior performance with less than half the computational cost during continual pre-training. Extensive analyses demonstrate its effectiveness across different model sizes and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13791v1">From Tools to Teammates: Evaluating LLMs in Multi-Session Coding Interactions</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in working environments for a wide range of tasks, excelling at solving individual problems in isolation. However, are they also able to effectively collaborate over long-term interactions? To investigate this, we introduce MemoryCode, a synthetic multi-session dataset designed to test LLMs' ability to track and execute simple coding instructions amid irrelevant information, simulating a realistic setting. While all the models we tested handle isolated instructions well, even the performance of state-of-the-art models like GPT-4o deteriorates when instructions are spread across sessions. Our analysis suggests this is due to their failure to retrieve and integrate information over long instruction chains. Our results highlight a fundamental limitation of current LLMs, restricting their ability to collaborate effectively in long interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13783v1">Generative Large Recommendation Models: Emerging Trends in LLMs for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 This paper has been accepted for the tutorial track at WWW 2025
    </div>
    <details class="paper-abstract">
      In the era of information overload, recommendation systems play a pivotal role in filtering data and delivering personalized content. Recent advancements in feature interaction and user behavior modeling have significantly enhanced the recall and ranking processes of these systems. With the rise of large language models (LLMs), new opportunities have emerged to further improve recommendation systems. This tutorial explores two primary approaches for integrating LLMs: LLMs-enhanced recommendations, which leverage the reasoning capabilities of general LLMs, and generative large recommendation models, which focus on scaling and sophistication. While the former has been extensively covered in existing literature, the latter remains underexplored. This tutorial aims to fill this gap by providing a comprehensive overview of generative large recommendation models, including their recent advancements, challenges, and potential research directions. Key topics include data quality, scaling laws, user behavior mining, and efficiency in training and inference. By engaging with this tutorial, participants will gain insights into the latest developments and future opportunities in the field, aiding both academic research and practical applications. The timely nature of this exploration supports the rapid evolution of recommendation systems, offering valuable guidance for researchers and practitioners alike.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13691v1">Is This Collection Worth My LLM's Time? Automatically Measuring Information Potential in Text Corpora</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) converge towards similar capabilities, the key to advancing their performance lies in identifying and incorporating valuable new information sources. However, evaluating which text collections are worth the substantial investment required for digitization, preprocessing, and integration into LLM systems remains a significant challenge. We present a novel approach to this challenge: an automated pipeline that evaluates the potential information gain from text collections without requiring model training or fine-tuning. Our method generates multiple choice questions (MCQs) from texts and measures an LLM's performance both with and without access to the source material. The performance gap between these conditions serves as a proxy for the collection's information potential. We validate our approach using three strategically selected datasets: EPFL PhD manuscripts (likely containing novel specialized knowledge), Wikipedia articles (presumably part of training data), and a synthetic baseline dataset. Our results demonstrate that this method effectively identifies collections containing valuable novel information, providing a practical tool for prioritizing data acquisition and integration efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13681v1">An LLM-based Agent for Reliable Docker Environment Configuration</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Environment configuration is a critical yet time-consuming step in software development, especially when dealing with unfamiliar code repositories. While Large Language Models (LLMs) demonstrate the potential to accomplish software engineering tasks, existing methods for environment configuration often rely on manual efforts or fragile scripts, leading to inefficiencies and unreliable outcomes. We introduce Repo2Run, the first LLM-based agent designed to fully automate environment configuration and generate executable Dockerfiles for arbitrary Python repositories. We address two major challenges: (1) enabling the LLM agent to configure environments within isolated Docker containers, and (2) ensuring the successful configuration process is recorded and accurately transferred to a Dockerfile without error. To achieve this, we propose atomic configuration synthesis, featuring a dual-environment architecture (internal and external environment) with a rollback mechanism to prevent environment "pollution" from failed commands, guaranteeing atomic execution (execute fully or not at all) and a Dockerfile generator to transfer successful configuration steps into runnable Dockerfiles. We evaluate Repo2Run~on our proposed benchmark of 420 recent Python repositories with unit tests, where it achieves an 86.0% success rate, outperforming the best baseline by 63.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17003v4">Safety Layers in Aligned Large Language Models: The Key to LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Aligned LLMs are secure, capable of recognizing and refusing to answer malicious questions. However, the role of internal parameters in maintaining such security is not well understood yet, further these models can be vulnerable to security degradation when subjected to fine-tuning attacks. To address these challenges, our work uncovers the mechanism behind security in aligned LLMs at the parameter level, identifying a small set of contiguous layers in the middle of the model that are crucial for distinguishing malicious queries from normal ones, referred to as ``safety layers". We first confirm the existence of these safety layers by analyzing variations in input vectors within the model's internal layers. Additionally, we leverage the over-rejection phenomenon and parameters scaling analysis to precisely locate the safety layers. Building on these findings, we propose a novel fine-tuning approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient of the safety layers during fine-tuning to address the security degradation. Our experiments demonstrate that the proposed approach can significantly preserve LLM security while maintaining performance and reducing computational resources compared to full fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13648v1">Reliability Across Parametric and External Knowledge: Understanding Knowledge Handling in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 under-review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enhance their problem-solving capability by leveraging both parametric and external knowledge. Beyond leveraging external knowledge to improve response accuracy, they require key capabilities for reliable knowledge-handling: resolving conflicts between knowledge sources, avoiding distraction from uninformative external knowledge, and abstaining when sufficient knowledge is unavailable. Prior studies have examined these scenarios in isolation or with limited scope. To systematically evaluate these capabilities, we introduce a comprehensive framework for analyzing knowledge-handling based on two key dimensions: the presence of parametric knowledge and the informativeness of external knowledge. Through analysis, we identify biases in knowledge utilization and examine how the ability to handle one scenario impacts performance in others. Furthermore, we demonstrate that training on data constructed based on the knowledge-handling scenarios improves LLMs' reliability in integrating and utilizing knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13640v1">Qorgau: Evaluating LLM Safety in Kazakh-Russian Bilingual Contexts</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to have the potential to generate harmful content, posing risks to users. While significant progress has been made in developing taxonomies for LLM risks and safety evaluation prompts, most studies have focused on monolingual contexts, primarily in English. However, language- and region-specific risks in bilingual contexts are often overlooked, and core findings can diverge from those in monolingual settings. In this paper, we introduce Qorgau, a novel dataset specifically designed for safety evaluation in Kazakh and Russian, reflecting the unique bilingual context in Kazakhstan, where both Kazakh (a low-resource language) and Russian (a high-resource language) are spoken. Experiments with both multilingual and language-specific LLMs reveal notable differences in safety performance, emphasizing the need for tailored, region-specific datasets to ensure the responsible and safe deployment of LLMs in countries like Kazakhstan. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13632v1">Concept Layers: Enhancing Interpretability and Intervenability via LLM Conceptualization</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The opaque nature of Large Language Models (LLMs) has led to significant research efforts aimed at enhancing their interpretability, primarily through post-hoc methods. More recent in-hoc approaches, such as Concept Bottleneck Models (CBMs), offer both interpretability and intervenability by incorporating explicit concept representations. However, these methods suffer from key limitations, including reliance on labeled concept datasets and significant architectural modifications that challenges re-integration into existing system pipelines. In this work, we introduce a new methodology for incorporating interpretability and intervenability into an existing model by integrating Concept Layers (CLs) into its architecture. Our approach projects the model's internal vector representations into a conceptual, explainable vector space before reconstructing and feeding them back into the model. Furthermore, we eliminate the need for a human-selected concept set by algorithmically searching an ontology for a set of concepts that can be either task-specific or task-agnostic. We evaluate CLs across multiple tasks, demonstrating that they maintain the original model's performance and agreement while enabling meaningful interventions. Additionally, we present a proof of concept showcasing an intervenability interface, allowing users to adjust model behavior dynamically, such as mitigating biases during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22118v2">The Impact of Inference Acceleration on Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Last few years have seen unprecedented advances in capabilities of Large Language Models (LLMs). These advancements promise to benefit a vast array of application domains. However, due to their immense size, performing inference with LLMs is both costly and slow. Consequently, a plethora of recent work has proposed strategies to enhance inference efficiency, e.g., quantization, pruning, and caching. These acceleration strategies reduce the inference cost and latency, often by several factors, while maintaining much of the predictive performance measured via common benchmarks. In this work, we explore another critical aspect of LLM performance: demographic bias in model generations due to inference acceleration optimizations. Using a wide range of metrics, we probe bias in model outputs from a number of angles. Analysis of outputs before and after inference acceleration shows significant change in bias. Worryingly, these bias effects are complex and unpredictable. A combination of an acceleration strategy and bias type may show little bias change in one model but may lead to a large effect in another. Our results highlight a need for in-depth and case-by-case evaluation of model bias after it has been modified to accelerate inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13606v1">LaVCa: LLM-assisted Visual Cortex Captioning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      Understanding the property of neural populations (or voxels) in the human brain can advance our comprehension of human perceptual and cognitive processing capabilities and contribute to developing brain-inspired computer models. Recent encoding models using deep neural networks (DNNs) have successfully predicted voxel-wise activity. However, interpreting the properties that explain voxel responses remains challenging because of the black-box nature of DNNs. As a solution, we propose LLM-assisted Visual Cortex Captioning (LaVCa), a data-driven approach that uses large language models (LLMs) to generate natural-language captions for images to which voxels are selective. By applying LaVCa for image-evoked brain activity, we demonstrate that LaVCa generates captions that describe voxel selectivity more accurately than the previously proposed method. Furthermore, the captions generated by LaVCa quantitatively capture more detailed properties than the existing method at both the inter-voxel and intra-voxel levels. Furthermore, a more detailed analysis of the voxel-specific properties generated by LaVCa reveals fine-grained functional differentiation within regions of interest (ROIs) in the visual cortex and voxels that simultaneously represent multiple distinct concepts. These findings offer profound insights into human visual representations by assigning detailed captions throughout the visual cortex while highlighting the potential of LLM-based methods in understanding brain representations. Please check out our webpage at https://sites.google.com/view/lavca-llm/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13603v1">Efficient Safety Retrofitting Against Jailbreaking for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) is an efficient alignment technique that steers LLMs towards preferable outputs by training on preference data, bypassing the need for explicit reward models. Its simplicity enables easy adaptation to various domains and safety requirements. This paper examines DPO's effectiveness in model safety against jailbreaking attacks while minimizing data requirements and training costs. We introduce Egida, a dataset expanded from multiple sources, which includes 27 different safety topics and 18 different attack styles, complemented with synthetic and human labels. This data is used to boost the safety of state-of-the-art LLMs (Llama-3.1-8B/70B-Instruct, Qwen-2.5-7B/72B-Instruct) across topics and attack styles. In addition to safety evaluations, we assess their post-alignment performance degradation in general purpose tasks, and their tendency to over refusal. Following the proposed methodology, trained models reduce their Attack Success Rate by 10%-30%, using small training efforts (2,000 samples) with low computational cost (3\$ for 8B models, 20\$ for 72B models). Safety aligned models generalize to unseen topics and attack styles, with the most successful attack style reaching a success rate around 5%. Size and family are found to strongly influence model malleability towards safety, pointing at the importance of pre-training choices. To validate our findings, a large independent assessment of human preference agreement with Llama-Guard-3-8B is conducted by the authors and the associated dataset Egida-HSafe is released. Overall, this study illustrates how affordable and accessible it is to enhance LLM safety using DPO while outlining its current limitations. All datasets and models are released to enable reproducibility and further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13577v1">Unraveling the Localized Latents: Learning Stratified Manifold Structures in LLM Embedding Space with Sparse Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      However, real-world data often exhibit complex local structures that can be challenging for single-model approaches with a smooth global manifold in the embedding space to unravel. In this work, we conjecture that in the latent space of these large language models, the embeddings live in a local manifold structure with different dimensions depending on the perplexities and domains of the input data, commonly referred to as a Stratified Manifold structure, which in combination form a structured space known as a Stratified Space. To investigate the validity of this structural claim, we propose an analysis framework based on a Mixture-of-Experts (MoE) model where each expert is implemented with a simple dictionary learning algorithm at varying sparsity levels. By incorporating an attention-based soft-gating network, we verify that our model learns specialized sub-manifolds for an ensemble of input data sources, reflecting the semantic stratification in LLM embedding space. We further analyze the intrinsic dimensions of these stratified sub-manifolds and present extensive statistics on expert assignments, gating entropy, and inter-expert distances. Our experimental results demonstrate that our method not only validates the claim of a stratified manifold structure in the LLM embedding space, but also provides interpretable clusters that align with the intrinsic semantic variations of the input data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13542v1">Activation-aware Probe-Query: Effective Key-Value Retrieval for Long-Context LLMs Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have showcased exceptional performance in long-context tasks, while facing significant inference efficiency challenges with limited GPU memory. Existing solutions first proposed the sliding-window approach to accumulate a set of historical \textbf{key-value} (KV) pairs for reuse, then further improvements selectively retain its subsets at each step. However, due to the sparse attention distribution across a long context, it is hard to identify and recall relevant KV pairs, as the attention is distracted by massive candidate pairs. Additionally, we found it promising to select representative tokens as probe-Query in each sliding window to effectively represent the entire context, which is an approach overlooked by existing methods. Thus, we propose \textbf{ActQKV}, a training-free, \textbf{Act}ivation-aware approach that dynamically determines probe-\textbf{Q}uery and leverages it to retrieve the relevant \textbf{KV} pairs for inference. Specifically, ActQKV monitors a token-level indicator, Activation Bias, within each context window, enabling the proper construction of probe-Query for retrieval at pre-filling stage. To accurately recall the relevant KV pairs and minimize the irrelevant ones, we design a dynamic KV cut-off mechanism guided by information density across layers at the decoding stage. Experiments on the Long-Bench and $\infty$ Benchmarks demonstrate its state-of-the-art performance with competitive inference quality and resource efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08904v2">MIH-TCCT: Mitigating Inconsistent Hallucinations in LLMs via Event-Driven Text-Code Cyclic Training</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent methodologies utilizing synthetic datasets have aimed to address inconsistent hallucinations in large language models (LLMs); however,these approaches are primarily tailored to specific tasks, limiting their generalizability. Inspired by the strong performance of code-trained models in logic-intensive domains, we propose a novel framework that leverages event-based text to generate corresponding code and employs cyclic training to transfer the logical consistency of code to natural language effectively. Our method significantly reduces inconsistent hallucinations across three leading LLMs and two categories of natural language tasks while maintaining overall performance. This framework effectively alleviates hallucinations without necessitating adaptation to downstream tasks, demonstrating generality and providing new perspectives to tackle the challenge of inconsistent hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13502v1">PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 15 pages, 1 figure, 12 tables
    </div>
    <details class="paper-abstract">
      We show that Large Language Model from Power Law Decoder Representations (PLDR-LLM) is a foundational model whose deductive outputs are invariant tensors up to a small perturbation. PLDR-LLM learns a singularity condition for the deductive outputs that enable the once-inferred energy-curvature tensor $\mathbf{G}_{LM}$ to replace the deep neural network of power law graph attention (PLGA) generating the deductive outputs at inference. We demonstrate that a cache for $\mathbf{G}_{LM}$ (G-cache) and KV-cache can be implemented in a straightforward manner to improve the inference time. The invariance and generalizable nature of deductive outputs is at a very high fidelity where deductive outputs have same RMSE and determinant values up to 15 decimal places after caching, and zero-shot benchmark scores remain unchanged. Ablation studies show that learned deductive outputs have distinct loss and accuracy characteristics from models pretrained with transferred, randomly initialized or identity tensors as a constant tensor operator and an LLM with scaled-dot product attention (SDPA) is a special case of PLDR-LLM where $\mathbf{G}_{LM}$ is predefined as identity. The observed invariance characteristic introduces a novel asymmetry between training and inference phases with caching. We outline observed common characteristics of the deductive outputs for the learned singularity condition. We provide an implementation of a training and inference framework for PLDR-LLM with KV-cache and G-cache.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11506v2">Glimpse: Enabling White-Box Methods to Use Proprietary Models for Zero-Shot LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 ICLR 2025 camera version (10 pages, 9 figures, 9 tables)
    </div>
    <details class="paper-abstract">
      Advanced large language models (LLMs) can generate text almost indistinguishable from human-written text, highlighting the importance of LLM-generated text detection. However, current zero-shot techniques face challenges as white-box methods are restricted to use weaker open-source LLMs, and black-box methods are limited by partial observation from stronger proprietary LLMs. It seems impossible to enable white-box methods to use proprietary models because API-level access to the models neither provides full predictive distributions nor inner embeddings. To traverse the divide, we propose **Glimpse**, a probability distribution estimation approach, predicting the full distributions from partial observations. Despite the simplicity of Glimpse, we successfully extend white-box methods like Entropy, Rank, Log-Rank, and Fast-DetectGPT to latest proprietary models. Experiments show that Glimpse with Fast-DetectGPT and GPT-3.5 achieves an average AUROC of about 0.95 in five latest source models, improving the score by 51% relative to the remaining space of the open source baseline. It demonstrates that the latest LLMs can effectively detect their own outputs, suggesting that advanced LLMs may be the best shield against themselves. We release our code and data at https://github.com/baoguangsheng/glimpse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13499v1">Hidden Darkness in LLM-Generated Designs: Exploring Dark Patterns in Ecommerce Web Components Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent work has highlighted the risks of LLM-generated content for a wide range of harmful behaviors, including incorrect and harmful code. In this work, we extend this by studying whether LLM-generated web design contains dark patterns. This work evaluated designs of ecommerce web components generated by four popular LLMs: Claude, GPT, Gemini, and Llama. We tested 13 commonly used ecommerce components (e.g., search, product reviews) and used them as prompts to generate a total of 312 components across all models. Over one-third of generated components contain at least one dark pattern. The majority of dark pattern strategies involve hiding crucial information, limiting users' actions, and manipulating them into making decisions through a sense of urgency. Dark patterns are also more frequently produced in components that are related to company interests. These findings highlight the need for interventions to prevent dark patterns during front-end code generation with LLMs and emphasize the importance of expanding ethical design education to a broader audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v1">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been demonstrated to have gaps in diverse, cultural knowledge across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on the ability of LLMs to display familiarity with a diverse range of national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on a series of cultural familiarity benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., the norms, artifacts, and institutions of national cultures), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models, while failing to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional knowledge about a culture and open-ended cultural fluency when it comes to evaluating the cultural familiarity of generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11149v2">Large Language-Geometry Model: When LLM meets Equivariance</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Accurately predicting 3D structures and dynamics of physical systems is crucial in scientific applications. Existing approaches that rely on geometric Graph Neural Networks (GNNs) effectively enforce $\mathrm{E}(3)$-equivariance, but they often fall in leveraging extensive broader information. While direct application of Large Language Models (LLMs) can incorporate external knowledge, they lack the capability for spatial reasoning with guaranteed equivariance. In this paper, we propose EquiLLM, a novel framework for representing 3D physical systems that seamlessly integrates E(3)-equivariance with LLM capabilities. Specifically, EquiLLM comprises four key components: geometry-aware prompting, an equivariant encoder, an LLM, and an equivariant adaptor. Essentially, the LLM guided by the instructive prompt serves as a sophisticated invariant feature processor, while 3D directional information is exclusively handled by the equivariant encoder and adaptor modules. Experimental results demonstrate that EquiLLM delivers significant improvements over previous methods across molecular dynamics simulation, human motion simulation, and antibody design, highlighting its promising generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09403v2">Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09282v2">CRVQ: Channel-Relaxed Vector Quantization for Extreme Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 7 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Powerful large language models (LLMs) are increasingly expected to be deployed with lower computational costs, enabling their capabilities on resource-constrained devices. Post-training quantization (PTQ) has emerged as a star approach to achieve this ambition, with best methods compressing weights to less than 2 bit on average. In this paper, we propose Channel-Relaxed Vector Quantization (CRVQ), a novel technique that significantly improves the performance of PTQ baselines at the cost of only minimal additional bits. This state-of-the-art extreme compression method achieves its results through two key innovations: (1) carefully selecting and reordering a very small subset of critical weight channels, and (2) leveraging extended codebooks to relax the constraint of critical channels. With our method, we demonstrate a 38.9\% improvement over the current strongest sub-2-bit PTQ baseline, enabling nearer lossless 1-bit compression. Furthermore, our approach offers flexible customization of quantization bit-width and performance, providing a wider range of deployment options for diverse hardware platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13442v1">TreeCut: A Synthetic Unanswerable Math Word Problem Dataset for LLM Hallucination Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now achieve near-human performance on standard math word problem benchmarks (e.g., GSM8K), yet their true reasoning ability remains disputed. A key concern is that models often produce confident, yet unfounded, answers to unanswerable problems. We introduce TreeCut, a synthetic dataset that systematically generates infinite unanswerable math word problems and their answerable counterparts, by representing each question as a tree and removing chosen necessary conditions. Experiments show TreeCut effectively induce hallucinations in large language models, including GPT-4o and o3-mini, with rates of 61% and 42% in their respective worst-case scenarios. Further analysis highlights that deeper or more complex trees, composite item names, and removing necessary condition near the middle of a path all increase the likelihood of hallucinations, underscoring the persistent challenges LLMs face in identifying unanswerable math problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.04047v3">AutoParLLM: GNN-guided Context Generation for Zero-Shot Code Parallelization using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      In-Context Learning (ICL) has been shown to be a powerful technique to augment the capabilities of LLMs for a diverse range of tasks. This work proposes \ourtool, a novel way to generate context using guidance from graph neural networks (GNNs) to generate efficient parallel codes. We evaluate \ourtool \xspace{} on $12$ applications from two well-known benchmark suites of parallel codes: NAS Parallel Benchmark and Rodinia Benchmark. Our results show that \ourtool \xspace{} improves the state-of-the-art LLMs (e.g., GPT-4) by 19.9\% in NAS and 6.48\% in Rodinia benchmark in terms of CodeBERTScore for the task of parallel code generation. Moreover, \ourtool \xspace{} improves the ability of the most powerful LLM to date, GPT-4, by achieving $\approx$17\% (on NAS benchmark) and $\approx$16\% (on Rodinia benchmark) better speedup. In addition, we propose \ourscore \xspace{} for evaluating the quality of the parallel code and show its effectiveness in evaluating parallel codes. \ourtool \xspace is available at https://github.com/quazirafi/AutoParLLM.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13417v1">RLTHF: Targeted Human Feedback for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to align with user preferences is challenging due to the high cost of quality human annotations in Reinforcement Learning from Human Feedback (RLHF) and the generalizability limitations of AI Feedback. To address these challenges, we propose RLTHF, a human-AI hybrid framework that combines LLM-based initial alignment with selective human annotations to achieve full-human annotation alignment with minimal effort. RLTHF identifies hard-to-annotate samples mislabeled by LLMs using a reward model's reward distribution and iteratively enhances alignment by integrating strategic human corrections while leveraging LLM's correctly labeled samples. Evaluations on HH-RLHF and TL;DR datasets show that RLTHF reaches full-human annotation-level alignment with only 6-7% of the human annotation effort. Furthermore, models trained on RLTHF's curated datasets for downstream tasks outperform those trained on fully human-annotated datasets, underscoring the effectiveness of RLTHF's strategic data curation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13416v1">Detecting LLM Fact-conflicting Hallucinations Enhanced by Temporal-logic-based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 16 pages, under review. arXiv admin note: substantial text overlap with arXiv:2405.00648
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face the challenge of hallucinations -- outputs that seem coherent but are actually incorrect. A particularly damaging type is fact-conflicting hallucination (FCH), where generated content contradicts established facts. Addressing FCH presents three main challenges: 1) Automatically constructing and maintaining large-scale benchmark datasets is difficult and resource-intensive; 2) Generating complex and efficient test cases that the LLM has not been trained on -- especially those involving intricate temporal features -- is challenging, yet crucial for eliciting hallucinations; and 3) Validating the reasoning behind LLM outputs is inherently difficult, particularly with complex logical relationships, as it requires transparency in the model's decision-making process. This paper presents Drowzee, an innovative end-to-end metamorphic testing framework that utilizes temporal logic to identify fact-conflicting hallucinations (FCH) in large language models (LLMs). Drowzee builds a comprehensive factual knowledge base by crawling sources like Wikipedia and uses automated temporal-logic reasoning to convert this knowledge into a large, extensible set of test cases with ground truth answers. LLMs are tested using these cases through template-based prompts, which require them to generate both answers and reasoning steps. To validate the reasoning, we propose two semantic-aware oracles that compare the semantic structure of LLM outputs to the ground truths. Across nine LLMs in nine different knowledge domains, experimental results show that Drowzee effectively identifies rates of non-temporal-related hallucinations ranging from 24.7% to 59.8%, and rates of temporal-related hallucinations ranging from 16.7% to 39.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06647v4">How Efficient is LLM-Generated Code? A Rigorous & High-Standard Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has significantly pushed the frontiers of program synthesis. Advancement of LLM-based program synthesis calls for a thorough evaluation of LLM-generated code. Most evaluation frameworks focus on the (functional) correctness of generated code; efficiency, as an important measure of code quality, has been overlooked in existing evaluations. In this work, we develop ENAMEL (EfficeNcy AutoMatic EvaLuator), a rigorous and high-standard benchmark for evaluating the capability of LLMs in generating efficient code. Firstly, we propose a new efficiency metric called eff@k, which generalizes the pass@k metric from correctness to efficiency and appropriately handles right-censored execution time. Furthermore, we derive an unbiased and variance-reduced estimator of eff@k via Rao--Blackwellization; we also provide a numerically stable implementation for the new estimator. Secondly, to set a high-standard for efficiency evaluation, we employ a human expert to design best algorithms and implementations as our reference solutions of efficiency, many of which are much more efficient than existing canonical solutions in HumanEval and HumanEval+. Moreover, to ensure a rigorous evaluation, we employ a human expert to curate strong test case generators to filter out wrong code and differentiate suboptimal algorithms. An extensive study across 30 popular LLMs using our benchmark ENAMEL shows that LLMs still fall short of generating expert-level efficient code. Using two subsets of our problem set, we demonstrate that such deficiency is because current LLMs struggle in designing advanced algorithms and are barely aware of implementation optimization. Our benchmark is publicly available at https://github.com/q-rz/enamel .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v2">RevPRAG: Revealing Poisoning Attacks in Retrieval-Augmented Generation through LLM Activation Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enriches the input to LLMs by retrieving information from the relevant knowledge database, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11433v3">FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08631v2">Ensemble based approach to quantifying uncertainty of LLM based classifications</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      The output of Large Language Models (LLMs) are a function of the internal model's parameters and the input provided into the context window. The hypothesis presented here is that under a greedy sampling strategy the variance in the LLM's output is a function of the conceptual certainty embedded in the model's parametric knowledge, as well as the lexical variance in the input. Finetuning the model results in reducing the sensitivity of the model output to the lexical input variations. This is then applied to a classification problem and a probabilistic method is proposed for estimating the certainties of the predicted classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13396v1">Prompting a Weighting Mechanism into LLM-as-a-Judge in Two-Step: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 5 pages, 5 tables, 1 figure
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have emerged as promising tools for evaluating Natural Language Generation (NLG) tasks, their effectiveness is limited by their inability to appropriately weigh the importance of different topics, often overemphasizing minor details while undervaluing critical information, leading to misleading assessments. Our work proposes an efficient prompt design mechanism to address this specific limitation and provide a case study. Through strategic prompt engineering that incorporates explicit importance weighting mechanisms, we enhance using LLM-as-a-Judge ability to prioritize relevant information effectively, as demonstrated by an average improvement of 6% in the Human Alignment Rate (HAR) metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10844v2">Be Friendly, Not Friends: How LLM Sycophancy Shapes User Trust</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that large language model (LLM)-powered conversational agents often exhibit `sycophancy', a tendency to adapt their responses to align with user perspectives, even at the expense of factual accuracy. However, users' perceptions of LLM sycophancy and its interplay with other anthropomorphic features (e.g., friendliness) in shaping user trust remains understudied. To bridge this gap, we conducted a 2 (Sycophancy: presence vs. absence) x 2 (Friendliness: high vs. low) between-subjects experiment (N = 224). Our study uncovered, for the first time, the intricate dynamics between LLM sycophancy and friendliness: When an LLM agent already exhibits a friendly demeanor, being sycophantic reduces perceived authenticity, thereby lowering user trust; Conversely, when the agent is less friendly, aligning its responses with user opinions makes it appear more genuine, leading to higher user trust. Our findings entail profound implications for AI persuasion through exploiting human psychological tendencies and highlight the imperative for responsible designs in user-LLM agent interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12945v2">LLMPopcorn: An Empirical Study of LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Popular Micro-videos, dominant on platforms like TikTok and YouTube, hold significant commercial value. The rise of high-quality AI-generated content has spurred interest in AI-driven micro-video creation. However, despite the advanced capabilities of large language models (LLMs) like ChatGPT and DeepSeek in text generation and reasoning, their potential to assist the creation of popular micro-videos remains largely unexplored. In this paper, we conduct an empirical study on LLM-assisted popular micro-video generation (LLMPopcorn). Specifically, we investigate the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? By exploring these questions, we show that advanced LLMs like DeepSeek-V3 enable micro-video generation to achieve popularity comparable to human-created content. Prompt enhancements further boost popularity, and benchmarking highlights DeepSeek-V3 and DeepSeek-R1 among LLMs, while LTX-Video and HunyuanVideo lead in video generation. This pioneering work advances AI-assisted micro-video creation, uncovering new research opportunities. We will release the code and datasets to support future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13358v1">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, yet they still struggle with direct text editing tasks that demand precise, context-aware modifications. While models like ChatGPT excel in text generation and analysis, their editing abilities often fall short, addressing only superficial issues rather than deeper structural or logical inconsistencies. In this work, we introduce a dual approach to enhance LLMs editing performance. First, we present InstrEditBench, a high-quality benchmark dataset comprising over 20,000 structured editing tasks spanning Wiki articles, LaTeX documents, code, and database Domain-specific Languages (DSL). InstrEditBench is generated using an innovative automated workflow that accurately identifies and evaluates targeted edits, ensuring that modifications adhere strictly to specified instructions without altering unrelated content. Second, we propose FineEdit, a specialized model trained on this curated benchmark. Experimental results demonstrate that FineEdit achieves significant improvements around {10\%} compared with Gemini on direct editing tasks, convincingly validating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10937v3">Proxona: Supporting Creators' Sensemaking and Ideation with LLM-Powered Audience Personas</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Accepted by ACM CHI 2025; 32 pages (including 11 pages of Appendix); Acknowledgment added
    </div>
    <details class="paper-abstract">
      A content creator's success depends on understanding their audience, but existing tools fail to provide in-depth insights and actionable feedback necessary for effectively targeting their audience. We present Proxona, an LLM-powered system that transforms static audience comments into interactive, multi-dimensional personas, allowing creators to engage with them to gain insights, gather simulated feedback, and refine content. Proxona distills audience traits from comments, into dimensions (categories) and values (attributes), then clusters them into interactive personas representing audience segments. Technical evaluations show that Proxona generates diverse dimensions and values, enabling the creation of personas that sufficiently reflect the audience and support data grounded conversation. User evaluation with 11 creators confirmed that Proxona helped creators discover hidden audiences, gain persona-informed insights on early-stage content, and allowed them to confidently employ strategies when iteratively creating storylines. Proxona introduces a novel creator-audience interaction framework and fosters a persona-driven, co-creative process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13347v1">Craw4LLM: Efficient Web Crawling for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Web crawl is a main source of large language models' (LLMs) pretraining data, but the majority of crawled web pages are discarded in pretraining due to low data quality. This paper presents Crawl4LLM, an efficient web crawling method that explores the web graph based on the preference of LLM pretraining. Specifically, it leverages the influence of a webpage in LLM pretraining as the priority score of the web crawler's scheduler, replacing the standard graph connectivity based priority. Our experiments on a web graph containing 900 million webpages from a commercial search engine's index demonstrate the efficiency of Crawl4LLM in obtaining high-quality pretraining data. With just 21% URLs crawled, LLMs pretrained on Crawl4LLM data reach the same downstream performances of previous crawls, significantly reducing the crawling waste and alleviating the burdens on websites. Our code is publicly available at https://github.com/cxcscmu/Crawl4LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14145v1">LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 In submission to INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06556v3">ProjectTest: A Project-level LLM Unit Test Generation Benchmark and Impact of Error Fixing Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-02-19
    </div>
    <details class="paper-abstract">
      Unit test generation has become a promising and important use case of LLMs. However, existing evaluation benchmarks for assessing LLM unit test generation capabilities focus on function- or class-level code rather than more practical and challenging project-level codebases. To address such limitation, we propose ProjectTest, a project-level benchmark for unit test generation covering Python, Java, and JavaScript. ProjectTest features 20 moderate-sized and high-quality projects per language. We evaluate nine frontier LLMs on ProjectTest and the results show that all frontier LLMs tested exhibit moderate performance on ProjectTest on Python and Java, highlighting the difficulty of ProjectTest. We also conduct a thorough error analysis, which shows that even frontier LLMs, such as Claude-3.5-Sonnet, have significant basic yet critical errors, including compilation and cascade errors. Motivated by this observation, we further evaluate all frontier LLMs under manual error-fixing and self-error-fixing scenarios to assess their potential when equipped with error-fixing mechanisms. Our code and dataset is available at \href{https://github.com/YiboWANG214/ProjectTest}{ProjectTest}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10508v2">Prompt Engineering or Fine-Tuning: An Empirical Assessment of LLMs for Code</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 11 pages + reference. Accepted in 22nd International Conference on Mining Software Repositories, 2025. Technical Papers Track (MSR'25)
    </div>
    <details class="paper-abstract">
      The rapid advancements in large language models (LLMs) have greatly expanded the potential for automated code-related tasks. Two primary methodologies are used in this domain: prompt engineering and fine-tuning. Prompt engineering involves applying different strategies to query LLMs, like ChatGPT, while fine-tuning further adapts pre-trained models, such as CodeBERT, by training them on task-specific data. Despite the growth in the area, there remains a lack of comprehensive comparative analysis between the approaches for code models. In this paper, we evaluate GPT-4 using three prompt engineering strategies -- basic prompting, in-context learning, and task-specific prompting -- and compare it against 17 fine-tuned models across three code-related tasks: code summarization, generation, and translation. Our results indicate that GPT-4 with prompt engineering does not consistently outperform fine-tuned models. For instance, in code generation, GPT-4 is outperformed by fine-tuned models by 28.3% points on the MBPP dataset. It also shows mixed results for code translation tasks. Additionally, a user study was conducted involving 27 graduate students and 10 industry practitioners. The study revealed that GPT-4 with conversational prompts, incorporating human feedback during interaction, significantly improved performance compared to automated prompting. Participants often provided explicit instructions or added context during these interactions. These findings suggest that GPT-4 with conversational prompting holds significant promise for automated code-related tasks, whereas fully automated prompt engineering without human involvement still requires further investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14133v1">Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Pre-print, 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impacts of these unintended features toward classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed framework can significantly improve the classifier's generalizability by regularizing those features that are not semantically correlated to each task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. We will release our code and data once accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12003v2">RAG-Optimized Tibetan Tourism LLMs: Enhancing Accuracy and Personalization</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 Accepted by AIPR 2024
    </div>
    <details class="paper-abstract">
      With the development of the modern social economy, tourism has become an important way to meet people's spiritual needs, bringing development opportunities to the tourism industry. However, existing large language models (LLMs) face challenges in personalized recommendation capabilities and the generation of content that can sometimes produce hallucinations. This study proposes an optimization scheme for Tibet tourism LLMs based on retrieval-augmented generation (RAG) technology. By constructing a database of tourist viewpoints and processing the data using vectorization techniques, we have significantly improved retrieval accuracy. The application of RAG technology effectively addresses the hallucination problem in content generation. The optimized model shows significant improvements in fluency, accuracy, and relevance of content generation. This research demonstrates the potential of RAG technology in the standardization of cultural tourism information and data analysis, providing theoretical and technical support for the development of intelligent cultural tourism service systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14127v1">Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above</a></div>
    <div class="paper-meta">
      📅 2025-02-19
      | 💬 In-progress preprint
    </div>
    <details class="paper-abstract">
      Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing-where LLMs construct and explain answers-better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA-robustness, biases, and unfaithful explanations-showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12583v1">LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Despite the growing development of long-context large language models (LLMs), data-centric approaches relying on synthetic data have been hindered by issues related to faithfulness, which limit their effectiveness in enhancing model performance on tasks such as long-context reasoning and question answering (QA). These challenges are often exacerbated by misinformation caused by lack of verification, reasoning without attribution, and potential knowledge conflicts. We propose LongFaith, a novel pipeline for synthesizing faithful long-context reasoning instruction datasets. By integrating ground truth and citation-based reasoning prompts, we eliminate distractions and improve the accuracy of reasoning chains, thus mitigating the need for costly verification processes. We open-source two synthesized datasets, LongFaith-SFT and LongFaith-PO, which systematically address multiple dimensions of faithfulness, including verified reasoning, attribution, and contextual grounding. Extensive experiments on multi-hop reasoning datasets and LongBench demonstrate that models fine-tuned on these datasets significantly improve performance. Our ablation studies highlight the scalability and adaptability of the LongFaith pipeline, showcasing its broad applicability in developing long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12575v1">DemonAgent: Dynamically Encrypted Multi-Backdoor Implantation Attack on LLM-based Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      As LLM-based agents become increasingly prevalent, backdoors can be implanted into agents through user queries or environment feedback, raising critical concerns regarding safety vulnerabilities. However, backdoor attacks are typically detectable by safety audits that analyze the reasoning process of agents. To this end, we propose a novel backdoor implantation strategy called \textbf{Dynamically Encrypted Multi-Backdoor Implantation Attack}. Specifically, we introduce dynamic encryption, which maps the backdoor into benign content, effectively circumventing safety audits. To enhance stealthiness, we further decompose the backdoor into multiple sub-backdoor fragments. Based on these advancements, backdoors are allowed to bypass safety audits significantly. Additionally, we present AgentBackdoorEval, a dataset designed for the comprehensive evaluation of agent backdoor attacks. Experimental results across multiple datasets demonstrate that our method achieves an attack success rate nearing 100\% while maintaining a detection rate of 0\%, illustrating its effectiveness in evading safety audits. Our findings highlight the limitations of existing safety mechanisms in detecting advanced attacks, underscoring the urgent need for more robust defenses against backdoor threats. Code and data are available at https://github.com/whfeLingYu/DemonAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12574v1">HeadInfer: Memory-Efficient LLM Inference by Head-wise Offloading</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) demonstrate impressive performance in long context generation. Extending the context length has disproportionately shifted the memory footprint of LLMs during inference to the key-value cache (KV cache). In this paper, we propose HEADINFER, which offloads the KV cache to CPU RAM while avoiding the need to fully store the KV cache for any transformer layer on the GPU. HEADINFER employs a fine-grained, head-wise offloading strategy, maintaining only selective attention heads KV cache on the GPU while computing attention output dynamically. Through roofline analysis, we demonstrate that HEADINFER maintains computational efficiency while significantly reducing memory footprint. We evaluate HEADINFER on the Llama-3-8B model with a 1-million-token sequence, reducing the GPU memory footprint of the KV cache from 128 GB to 1 GB and the total GPU memory usage from 207 GB to 17 GB, achieving a 92% reduction compared to BF16 baseline inference. Notably, HEADINFER enables 4-million-token inference with an 8B model on a single consumer GPU with 24GB memory (e.g., NVIDIA RTX 4090) without approximation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13982v1">Benchmarking Automatic Speech Recognition coupled LLM Modules for Medical Diagnostics</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Natural Language Processing (NLP) and Voice Recognition agents are rapidly evolving healthcare by enabling efficient, accessible, and professional patient support while automating grunt work. This report serves as my self project wherein models finetuned on medical call recordings are analysed through a two-stage system: Automatic Speech Recognition (ASR) for speech transcription and a Large Language Model (LLM) for context-aware, professional responses. ASR, finetuned on phone call recordings provides generalised transcription of diverse patient speech over call, while the LLM matches transcribed text to medical diagnosis. A novel audio preprocessing strategy, is deployed to provide invariance to incoming recording/call data, laden with sufficient augmentation with noise/clipping to make the pipeline robust to the type of microphone and ambient conditions the patient might have while calling/recording.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14905v1">Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      In this paper, we address the challenge of enforcing strict schema adherence in large language model (LLM) generation by leveraging LLM reasoning capabilities. Building on the DeepSeek R1 reinforcement learning framework, our approach trains structured reasoning skills of a 1.5B parameter model through a novel pipeline that combines synthetic reasoning dataset construction with custom reward functions under Group Relative Policy Optimization (GRPO). Specifically, we first perform R1 reinforcement learning on a 20K sample unstructured-to-structured dataset, mirroring the original DeepSeek R1 methods, to establish core reasoning abilities. Subsequently, we performed supervised fine-tuning on a separate 10K reasoning sample dataset, focusing on refining schema adherence for downstream tasks. Despite the relatively modest training scope, requiring approximately 20 hours on an 8xH100 GPU cluster for GRPO training and 3 hours on 1xA100 for SFT, our model demonstrates robust performance in enforcing schema consistency. We compare our ThinkJSON approach against the original DeepSeek R1 (671B), distilled versions of DeepSeek R1 (Qwen-1.5B and Qwen-7B), and Gemini 2.0 Flash (70B), showcasing its effectiveness in real-world applications. Our results underscore the practical utility of a resource-efficient framework for schema-constrained text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05957v2">AutoAgent: A Fully-Automated and Zero-Code Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Code: https://github.com/HKUDS/AutoAgent
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) Agents have demonstrated remarkable capabilities in task automation and intelligent decision-making, driving the widespread adoption of agent development frameworks such as LangChain and AutoGen. However, these frameworks predominantly serve developers with extensive technical expertise - a significant limitation considering that only 0.03 % of the global population possesses the necessary programming skills. This stark accessibility gap raises a fundamental question: Can we enable everyone, regardless of technical background, to build their own LLM agents using natural language alone? To address this challenge, we introduce AutoAgent-a Fully-Automated and highly Self-Developing framework that enables users to create and deploy LLM agents through Natural Language Alone. Operating as an autonomous Agent Operating System, AutoAgent comprises four key components: i) Agentic System Utilities, ii) LLM-powered Actionable Engine, iii) Self-Managing File System, and iv) Self-Play Agent Customization module. This lightweight yet powerful system enables efficient and dynamic creation and modification of tools, agents, and workflows without coding requirements or manual intervention. Beyond its code-free agent development capabilities, AutoAgent also serves as a versatile multi-agent system for General AI Assistants. Comprehensive evaluations on the GAIA benchmark demonstrate AutoAgent's effectiveness in generalist multi-agent tasks, surpassing existing state-of-the-art methods. Furthermore, AutoAgent's Retrieval-Augmented Generation (RAG)-related capabilities have shown consistently superior performance compared to many alternative LLM-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13879v3">Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 22 pages, 8 figures, 11 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12566v1">Exploring the Impact of Personality Traits on LLM Bias and Toxicity</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      With the different roles that AI is expected to play in human life, imbuing large language models (LLMs) with different personalities has attracted increasing research interests. While the "personification" enhances human experiences of interactivity and adaptability of LLMs, it gives rise to critical concerns about content safety, particularly regarding bias, sentiment and toxicity of LLM generation. This study explores how assigning different personality traits to LLMs affects the toxicity and biases of their outputs. Leveraging the widely accepted HEXACO personality framework developed in social psychology, we design experimentally sound prompts to test three LLMs' performance on three toxic and bias benchmarks. The findings demonstrate the sensitivity of all three models to HEXACO personality traits and, more importantly, a consistent variation in the biases, negative sentiment and toxicity of their output. In particular, adjusting the levels of several personality traits can effectively reduce bias and toxicity in model performance, similar to humans' correlations between personality traits and toxic behaviors. The findings highlight the additional need to examine content safety besides the efficiency of training or fine-tuning methods for LLM personification. They also suggest a potential for the adjustment of personalities to be a simple and low-cost method to conduct controlled text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12561v1">UXAgent: An LLM Agent-Based Usability Testing Framework for Web Design</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Usability testing is a fundamental yet challenging (e.g., inflexible to iterate the study design flaws and hard to recruit study participants) research method for user experience (UX) researchers to evaluate a web design. Recent advances in Large Language Model-simulated Agent (LLM-Agent) research inspired us to design UXAgent to support UX researchers in evaluating and reiterating their usability testing study design before they conduct the real human subject study. Our system features an LLM-Agent module and a universal browser connector module so that UX researchers can automatically generate thousands of simulated users to test the target website. The results are shown in qualitative (e.g., interviewing how an agent thinks ), quantitative (e.g., # of actions), and video recording formats for UX researchers to analyze. Through a heuristic user evaluation with five UX researchers, participants praised the innovation of our system but also expressed concerns about the future of LLM Agent-assisted UX study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12559v1">Distributed On-Device LLM Inference With Over-the-Air Computation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across various artificial intelligence tasks. However, their enormous sizes and computational demands pose significant challenges for the deployment on edge devices. To address this issue, we present a distributed on-device LLM inference framework based on tensor parallelism, which partitions neural network tensors (e.g., weight matrices) of LLMs among multiple edge devices for collaborative inference. Nevertheless, tensor parallelism involves frequent all-reduce operations to aggregate intermediate layer outputs across participating devices during inference, resulting in substantial communication overhead. To mitigate this bottleneck, we propose an over-the-air computation method that leverages the analog superposition property of wireless multiple-access channels to facilitate fast all-reduce operations. To minimize the average transmission mean-squared error, we investigate joint model assignment and transceiver optimization, which can be formulated as a mixed-timescale stochastic non-convex optimization problem. Then, we develop a mixed-timescale algorithm leveraging semidefinite relaxation and stochastic successive convex approximation methods. Comprehensive simulation results will show that the proposed approach significantly reduces inference latency while improving accuracy. This makes distributed on-device LLM inference practical for resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02220v4">Data to Defense: The Role of Curation in Customizing LLMs Against Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely adapted for downstream applications through fine-tuning, a process named customization. However, recent studies have identified a vulnerability during this process, where malicious samples can compromise the robustness of LLMs and amplify harmful behaviors-an attack commonly referred to as jailbreaking. To address this challenge, we propose an adaptive data curation approach allowing any text to be curated to enhance its effectiveness in counteracting harmful samples during customization. To avoid the need for additional defensive modules, we further introduce a comprehensive mitigation framework spanning the lifecycle of the customization process: before customization to immunize LLMs against future jailbreak attempts, during customization to neutralize risks, and after customization to restore compromised models. Experimental results demonstrate a significant reduction in jailbreaking effects, achieving up to a 100% success rate in generating safe responses. By combining adaptive data curation with lifecycle-based mitigation strategies, this work represents a solid step forward in mitigating jailbreaking risks and ensuring the secure adaptation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12552v1">LLM Safety for Children</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      This paper analyzes the safety of Large Language Models (LLMs) in interactions with children below age of 18 years. Despite the transformative applications of LLMs in various aspects of children's lives such as education and therapy, there remains a significant gap in understanding and mitigating potential content harms specific to this demographic. The study acknowledges the diverse nature of children often overlooked by standard safety evaluations and proposes a comprehensive approach to evaluating LLM safety specifically for children. We list down potential risks that children may encounter when using LLM powered applications. Additionally we develop Child User Models that reflect the varied personalities and interests of children informed by literature in child care and psychology. These user models aim to bridge the existing gap in child safety literature across various fields. We utilize Child User Models to evaluate the safety of six state of the art LLMs. Our observations reveal significant safety gaps in LLMs particularly in categories harmful to children but not adults
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07374v2">LLMs Can Easily Learn to Reason from Demonstrations Structure, not content, is what matters!</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large reasoning models (LRMs) tackle complex reasoning problems by following long chain-of-thoughts (Long CoT) that incorporate reflection, backtracking, and self-validation. However, the training techniques and data requirements to elicit Long CoT remain poorly understood. In this work, we find that a Large Language model (LLM) can effectively learn Long CoT reasoning through data-efficient supervised fine-tuning (SFT) and parameter-efficient low-rank adaptation (LoRA). With just 17k long CoT training samples, the Qwen2.5-32B-Instruct model achieves significant improvements on a wide range of math and coding benchmarks, including 56.7% (+40.0%) on AIME 2024 and 57.0% (+8.1%) on LiveCodeBench, competitive to the proprietary o1-preview model's score of 44.6% and 59.1%. More importantly, we find that the structure of Long CoT is critical to the learning process, whereas the content of individual reasoning steps has minimal impact. Perturbations affecting content, such as training on incorrect samples or removing reasoning keywords, have little impact on performance. In contrast, structural modifications that disrupt logical consistency in the Long CoT, such as shuffling or deleting reasoning steps, significantly degrade accuracy. For example, a model trained on Long CoT samples with incorrect answers still achieves only 3.2% lower accuracy compared to training with fully correct samples. These insights deepen our understanding of how to elicit reasoning capabilities in LLMs and highlight key considerations for efficiently training the next generation of reasoning models. This is the academic paper of our previous released Sky-T1-32B-Preview model. Codes are available at https://github.com/NovaSky-AI/SkyThought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14073v2">LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have been deployed in various real-world settings, concerns about the harm they may propagate have grown. Various jailbreaking techniques have been developed to expose the vulnerabilities of these models and improve their safety. This work reveals that many state-of-the-art LLMs are vulnerable to malicious requests hidden behind scientific language. Specifically, our experiments with GPT4o, GPT4o-mini, GPT-4, LLama3-405B-Instruct, Llama3-70B-Instruct, Cohere, Gemini models demonstrate that, the models' biases and toxicity substantially increase when prompted with requests that deliberately misinterpret social science and psychological studies as evidence supporting the benefits of stereotypical biases. Alarmingly, these models can also be manipulated to generate fabricated scientific arguments claiming that biases are beneficial, which can be used by ill-intended actors to systematically jailbreak these strong LLMs. Our analysis studies various factors that contribute to the models' vulnerabilities to malicious requests in academic language. Mentioning author names and venues enhances the persuasiveness of models, and the bias scores increase as dialogues progress. Our findings call for a more careful investigation on the use of scientific data for training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12531v1">GSCE: A Prompt Framework with Enhanced Reasoning for Reliable LLM-driven Drone Control</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into robotic control, including drones, has the potential to revolutionize autonomous systems. Research studies have demonstrated that LLMs can be leveraged to support robotic operations. However, when facing tasks with complex reasoning, concerns and challenges are raised about the reliability of solutions produced by LLMs. In this paper, we propose a prompt framework with enhanced reasoning to enable reliable LLM-driven control for drones. Our framework consists of novel technical components designed using Guidelines, Skill APIs, Constraints, and Examples, namely GSCE. GSCE is featured by its reliable and constraint-compliant code generation. We performed thorough experiments using GSCE for the control of drones with a wide level of task complexities. Our experiment results demonstrate that GSCE can significantly improve task success rates and completeness compared to baseline approaches, highlighting its potential for reliable LLM-driven autonomous drone systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12530v1">Policy-to-Language: Train LLMs to Explain Decisions with Flow-Matching Generated Rewards</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      As humans increasingly share environments with diverse agents powered by RL, LLMs, and beyond, the ability to explain their policies in natural language will be vital for reliable coexistence. In this paper, we build a model-agnostic explanation generator based on an LLM. The technical novelty is that the rewards for training this LLM are generated by a generative flow matching model. This model has a specially designed structure with a hidden layer merged with an LLM to harness the linguistic cues of explanations into generating appropriate rewards. Experiments on both RL and LLM tasks demonstrate that our method can generate dense and effective rewards while saving on expensive human feedback; it thus enables effective explanations and even improves the accuracy of the decisions in original tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12521v1">Inference-Time Computations for LLM Reasoning and Planning: A Benchmark and Insights</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      We examine the reasoning and planning capabilities of large language models (LLMs) in solving complex tasks. Recent advances in inference-time techniques demonstrate the potential to enhance LLM reasoning without additional training by exploring intermediate steps during inference. Notably, OpenAI's o1 model shows promising performance through its novel use of multi-step reasoning and verification. Here, we explore how scaling inference-time techniques can improve reasoning and planning, focusing on understanding the tradeoff between computational cost and performance. To this end, we construct a comprehensive benchmark, known as Sys2Bench, and perform extensive experiments evaluating existing inference-time techniques on eleven diverse tasks across five categories, including arithmetic reasoning, logical reasoning, common sense reasoning, algorithmic reasoning, and planning. Our findings indicate that simply scaling inference-time computation has limitations, as no single inference-time technique consistently performs well across all reasoning and planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15427v2">OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Customizable role-playing in large language models (LLMs), also known as character generalization, is gaining increasing attention for its versatility and cost-efficiency in developing and deploying role-playing dialogue agents. This study explores a large-scale data synthesis approach to equip LLMs with character generalization capabilities. We begin by synthesizing large-scale character profiles using personas from Persona Hub and then explore two strategies: response rewriting and response generation, to create character-aligned instructional responses. To validate the effectiveness of our synthetic instruction tuning data for character generalization, we perform supervised fine-tuning (SFT) using the LLaMA-3 8B model. Our best-performing model strengthens the original LLaMA-3 8B Instruct model and achieves performance comparable to GPT-4o models on role-playing dialogue. We release our synthetic characters and instruction-tuning dialogues to support public research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12516v1">Can LLMs Extract Frame-Semantic Arguments?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Frame-semantic parsing is a critical task in natural language understanding, yet the ability of large language models (LLMs) to extract frame-semantic arguments remains underexplored. This paper presents a comprehensive evaluation of LLMs on frame-semantic argument identification, analyzing the impact of input representation formats, model architectures, and generalization to unseen and out-of-domain samples. Our experiments, spanning models from 0.5B to 78B parameters, reveal that JSON-based representations significantly enhance performance, and while larger models generally perform better, smaller models can achieve competitive results through fine-tuning. We also introduce a novel approach to frame identification leveraging predicted frame elements, achieving state-of-the-art performance on ambiguous targets. Despite strong generalization capabilities, our analysis finds that LLMs still struggle with out-of-domain data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12504v1">Simulating Cooperative Prosocial Behavior with Multi-Agent LLMs: Evidence and Mechanisms for AI Agents to Inform Policy Decisions</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Human prosocial cooperation is essential for our collective health, education, and welfare. However, designing social systems to maintain or incentivize prosocial behavior is challenging because people can act selfishly to maximize personal gain. This complex and unpredictable aspect of human behavior makes it difficult for policymakers to foresee the implications of their designs. Recently, multi-agent LLM systems have shown remarkable capabilities in simulating human-like behavior, and replicating some human lab experiments. This paper studies how well multi-agent systems can simulate prosocial human behavior, such as that seen in the public goods game (PGG), and whether multi-agent systems can exhibit ``unbounded actions'' seen outside the lab in real world scenarios. We find that multi-agent LLM systems successfully replicate human behavior from lab experiments of the public goods game with three experimental treatments - priming, transparency, and varying endowments. Beyond replicating existing experiments, we find that multi-agent LLM systems can replicate the expected human behavior when combining experimental treatments, even if no previous study combined those specific treatments. Lastly, we find that multi-agent systems can exhibit a rich set of unbounded actions that people do in the real world outside of the lab -- such as collaborating and even cheating. In sum, these studies are steps towards a future where LLMs can be used to inform policy decisions that encourage people to act in a prosocial manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12501v1">Crowd Comparative Reasoning: Unlocking Comprehensive Evaluations for LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge, which generates chain-of-thought (CoT) judgments, has become a widely adopted auto-evaluation method. However, its reliability is compromised by the CoT reasoning's inability to capture comprehensive and deeper details, often leading to incomplete outcomes. Existing methods mainly rely on majority voting or criteria expansion, which is insufficient to address the limitation in CoT. We propose Crowd-based Comparative Evaluation, which introduces additional crowd responses to compare with the candidate responses, thereby exposing deeper and more comprehensive details within the candidate responses. This process effectively guides LLM-as-a-Judge to provide a more detailed CoT judgment. Extensive experiments demonstrate that our approach enhances evaluation reliability, achieving an average accuracy gain of 6.7% across five benchmarks. Moreover, our method produces higher-quality CoTs that facilitate judge distillation and exhibit superior performance in rejection sampling for supervised fine-tuning (SFT), referred to as crowd rejection sampling, thereby enabling more efficient SFT. Our analysis confirms that CoTs generated by ours are more comprehensive and of higher quality, and evaluation accuracy improves as inference scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12494v1">EDGE: Efficient Data Selection for LLM Agents via Guideline Effectiveness</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities as AI agents. However, existing methods for enhancing LLM-agent abilities often lack a focus on data quality, leading to inefficiencies and suboptimal results in both fine-tuning and prompt engineering. To address this issue, we introduce EDGE, a novel approach for identifying informative samples without needing golden answers. We propose the Guideline Effectiveness (GE) metric, which selects challenging samples by measuring the impact of human-provided guidelines in multi-turn interaction tasks. A low GE score indicates that the human expertise required for a sample is missing from the guideline, making the sample more informative. By selecting samples with low GE scores, we can improve the efficiency and outcomes of both prompt engineering and fine-tuning processes for LLMs. Extensive experiments validate the performance of our method. Our method achieves competitive results on the HotpotQA and WebShop and datasets, requiring 75\% and 50\% less data, respectively, while outperforming existing methods. We also provide a fresh perspective on the data quality of LLM-agent fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v1">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12478v1">MSE-Adapter: A Lightweight Plugin Endowing LLMs with the Capability to Perform Multimodal Sentiment Analysis and Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Current Multimodal Sentiment Analysis (MSA) and Emotion Recognition in Conversations (ERC) methods based on pre-trained language models exhibit two primary limitations: 1) Once trained for MSA and ERC tasks, these pre-trained language models lose their original generalized capabilities. 2) They demand considerable computational resources. As the size of pre-trained language models continues to grow, training larger multimodal sentiment analysis models using previous approaches could result in unnecessary computational cost. In response to this challenge, we propose \textbf{M}ultimodal \textbf{S}entiment Analysis and \textbf{E}motion Recognition \textbf{Adapter} (MSE-Adapter), a lightweight and adaptable plugin. This plugin enables a large language model (LLM) to carry out MSA or ERC tasks with minimal computational overhead (only introduces approximately 2.6M to 2.8M trainable parameters upon the 6/7B models), while preserving the intrinsic capabilities of the LLM. In the MSE-Adapter, the Text-Guide-Mixer (TGM) module is introduced to establish explicit connections between non-textual and textual modalities through the Hadamard product. This allows non-textual modalities to better align with textual modalities at the feature level, promoting the generation of higher-quality pseudo tokens. Extensive experiments were conducted on four public English and Chinese datasets using consumer-grade GPUs and open-source LLMs (Qwen-1.8B, ChatGLM3-6B-base, and LLaMA2-7B) as the backbone. The results demonstrate the effectiveness of the proposed plugin. The code will be released on GitHub after a blind review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12476v1">CoCo-CoLa: Evaluating Language Adherence in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) develop cross-lingual abilities despite being trained on limited parallel data. However, they often struggle to generate responses in the intended language, favoring high-resource languages such as English. In this work, we introduce CoCo-CoLa (Correct Concept - Correct Language), a novel metric to evaluate language adherence in multilingual LLMs. Using fine-tuning experiments on a closed-book QA task across seven languages, we analyze how training in one language affects others' performance. Our findings reveal that multilingual models share task knowledge across languages but exhibit biases in the selection of output language. We identify language-specific layers, showing that final layers play a crucial role in determining output language. Accordingly, we propose a partial training strategy that selectively fine-tunes key layers, improving language adherence while significantly reducing computational cost. Our method achieves comparable or superior performance to full fine-tuning, particularly for low-resource languages, offering a more efficient multilingual adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12470v1">Reasoning on a Spectrum: Aligning LLMs to System 1 and System 2 Thinking</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning abilities, yet their reliance on structured step-by-step processing reveals a critical limitation. While human cognition fluidly adapts between intuitive, heuristic (System 1) and analytical, deliberative (System 2) reasoning depending on the context, LLMs lack this dynamic flexibility. This rigidity can lead to brittle and unreliable performance when faced with tasks that deviate from their trained patterns. To address this, we create a dataset of 2,000 samples with valid System 1 and System 2 answers, explicitly align LLMs with these reasoning styles, and evaluate their performance across reasoning benchmarks. Our results reveal an accuracy-efficiency trade-off: System 2-aligned models excel in arithmetic and symbolic reasoning, while System 1-aligned models perform better in commonsense tasks. A mechanistic analysis of model responses shows that System 1 models employ more definitive answers, whereas System 2 models demonstrate greater uncertainty. Interpolating between these extremes produces a monotonic transition in reasoning accuracy, preserving coherence. This work challenges the assumption that step-by-step reasoning is always optimal and highlights the need for adapting reasoning strategies based on task demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12468v1">MCTS-Judge: Test-Time Scaling in LLM-as-a-Judge for Code Correctness Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      The LLM-as-a-Judge paradigm shows promise for evaluating generative content but lacks reliability in reasoning-intensive scenarios, such as programming. Inspired by recent advances in reasoning models and shifts in scaling laws, we pioneer bringing test-time computation into LLM-as-a-Judge, proposing MCTS-Judge, a resource-efficient, System-2 thinking framework for code correctness evaluation. MCTS-Judge leverages Monte Carlo Tree Search (MCTS) to decompose problems into simpler, multi-perspective evaluations. Through a node-selection strategy that combines self-assessment based on historical actions in the current trajectory and the Upper Confidence Bound for Trees based on prior rollouts, MCTS-Judge balances global optimization and refinement of the current trajectory. We further designed a high-precision, unit-test-level reward mechanism to encourage the Large Language Model (LLM) to perform line-by-line analysis. Extensive experiments on three benchmarks and five LLMs demonstrate the effectiveness of MCTS-Judge, which improves the base model's accuracy from 41% to 80%, surpassing the o1-series models with 3x fewer tokens. Further evaluations validate the superiority of its reasoning trajectory in logic, analytics, thoroughness, and overall quality, while revealing the test-time scaling law of the LLM-as-a-Judge paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12462v1">Emulating Retrieval Augmented Generation via Prompt Engineering for Enhanced Long Context Comprehension in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      This paper addresses the challenge of comprehending very long contexts in Large Language Models (LLMs) by proposing a method that emulates Retrieval Augmented Generation (RAG) through specialized prompt engineering and chain-of-thought (CoT) reasoning. While recent LLMs support over 100,000 tokens in a single prompt, simply enlarging context windows has not guaranteed robust multi-hop reasoning when key details are scattered across massive input. Our approach treats the model as both the retriever and the reasoner: it first tags relevant segments within a long passage, then employs a stepwise CoT workflow to integrate these pieces of evidence. This single-pass method thereby reduces reliance on an external retriever, yet maintains focus on crucial segments. We evaluate our approach on selected tasks from BABILong, which interleaves standard bAbI QA problems with large amounts of distractor text. Compared to baseline (no retrieval) and naive RAG pipelines, our approach more accurately handles multi-fact questions such as object location tracking, counting, and indefinite knowledge. Furthermore, we analyze how prompt structure, including the order of question, relevant-text tags, and overall instructions, significantly affects performance. These findings underscore that optimized prompt engineering, combined with guided reasoning, can enhance LLMs' long-context comprehension and serve as a lightweight alternative to traditional retrieval pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12460v1">LMN: A Tool for Generating Machine Enforceable Policies from Natural Language Access Control Rules using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Organizations often lay down rules or guidelines called Natural Language Access Control Policies (NLACPs) for specifying who gets access to which information and when. However, these cannot be directly used in a target access control model like Attribute-based Access Control (ABAC). Manually translating the NLACP rules into Machine Enforceable Security Policies (MESPs) is both time consuming and resource intensive, rendering it infeasible especially for large organizations. Automated machine translation workflows, on the other hand, require information security officers to be adept at using such processes. To effectively address this problem, we have developed a free web-based publicly accessible tool called LMN (LLMs for generating MESPs from NLACPs) that takes an NLACP as input and converts it into a corresponding MESP. Internally, LMN uses the GPT 3.5 API calls and an appropriately chosen prompt. Extensive experiments with different prompts and performance metrics firmly establish the usefulness of LMN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11433v2">FLAG-Trader: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12444v1">SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models have high compute, latency, and memory requirements. While specialized accelerators such as GPUs and TPUs typically run these workloads, CPUs are more widely available and consume less energy. Accelerating LLMs with CPUs enables broader AI access at a lower cost and power consumption. This acceleration potential for CPUs is especially relevant during the memory-bound decoding stage of LLM inference, which processes one token at a time and is becoming increasingly utilized with reasoning models. We utilize Advanced Matrix Extensions (AMX) support on the latest Intel CPUs together with unstructured sparsity to achieve a $1.42 \times$ reduction in end-to-end latency compared to the current PyTorch implementation by applying our technique in linear layers. We provide a set of open-source customized sparse kernels that can speed up any PyTorch model by automatically replacing all linear layers with our custom sparse implementation. Furthermore, we demonstrate for the first time the use of unstructured sparsity in the attention computation achieving a $1.14 \times$ speedup over the current systems without compromising accuracy. Code: https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/tree/main/SparAMX
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12126v2">What Do LLMs Need to Understand Graphs: A Survey of Parametric Representation of Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Preprint, 9 pages
    </div>
    <details class="paper-abstract">
      Graphs, as a relational data structure, have been widely used for various application scenarios, like molecule design and recommender systems. Recently, large language models (LLMs) are reorganizing in the AI community for their expected reasoning and inference abilities. Making LLMs understand graph-based relational data has great potential, including but not limited to (1) distillate external knowledge base for eliminating hallucination and breaking the context window limit for LLMs' inference during the retrieval augmentation generation process; (2) taking graph data as the input and directly solve the graph-based research tasks like protein design and drug discovery. However, inputting the entire graph data to LLMs is not practical due to its complex topological structure, data size, and the lack of effective and efficient semantic graph representations. A natural question arises: Is there a kind of graph representation that can be described by natural language for LLM's understanding and is also easy to require to serve as the raw input for LLMs? Based on statistical computation, graph laws pre-define a set of parameters (e.g., degree, time, diameter) and identifie their relationships and values by observing the topological distribution of plenty of real-world graph data. We believe this kind of parametric representation of graphs, graph laws, can be a solution for making LLMs understand graph data as the input. In this survey, we first review the previous study of graph laws from multiple perspectives, i.e., macroscope and microscope of graphs, low-order and high-order graphs, static and dynamic graphs, different observation spaces, and newly proposed graph parameters. After we review various real-world applications benefiting from the guidance of graph laws, we conclude the paper with current challenges and future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05002v5">An Empirical Study on Challenges for LLM Application Developers</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Accepted by ACM Transactions on Software Engineering and Methodology
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have seen rapid advancements, significantly impacting various fields such as computer vision, natural language processing, and software engineering. These LLMs, exemplified by OpenAI's ChatGPT, have revolutionized the way we approach language understanding and generation tasks. However, in contrast to traditional software development practices, LLM development introduces new challenges for AI developers in design, implementation, and deployment. These challenges span different areas (such as prompts, APIs, and plugins), requiring developers to navigate unique methodologies and considerations specific to LLM application development. Despite the profound influence of LLMs, to the best of our knowledge, these challenges have not been thoroughly investigated in previous empirical studies. To fill this gap, we present the first comprehensive study on understanding the challenges faced by LLM developers. Specifically, we crawl and analyze 29,057 relevant questions from a popular OpenAI developer forum. We first examine their popularity and difficulty. After manually analyzing 2,364 sampled questions, we construct a taxonomy of challenges faced by LLM developers. Based on this taxonomy, we summarize a set of findings and actionable implications for LLM-related stakeholders, including developers and providers (especially the OpenAI organization).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13319v1">Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      We know from prior work that LLMs encode social biases, and that this manifests in clinical tasks. In this work we adopt tools from mechanistic interpretability to unveil sociodemographic representations and biases within LLMs in the context of healthcare. Specifically, we ask: Can we identify activations within LLMs that encode sociodemographic information (e.g., gender, race)? We find that gender information is highly localized in middle MLP layers and can be reliably manipulated at inference time via patching. Such interventions can surgically alter generated clinical vignettes for specific conditions, and also influence downstream clinical predictions which correlate with gender, e.g., patient risk of depression. We find that representation of patient race is somewhat more distributed, but can also be intervened upon, to a degree. To our knowledge, this is the first application of mechanistic interpretability methods to LLMs for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13311v1">Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Intelligent tutoring agents powered by large language models (LLMs) have been increasingly explored to deliver personalized guidance in areas such as language learning and science education. However, their capabilities in guiding users to solve complex real-world tasks remain underexplored. To address this limitation, in this work, we focus on coding tutoring, a challenging problem that requires tutors to proactively guide students toward completing predefined coding tasks. We propose a novel agent workflow, Trace-and-Verify (TRAVER), which combines knowledge tracing to estimate a student's knowledge state and turn-by-turn verification to ensure effective guidance toward task completion. We introduce DICT, an automatic evaluation protocol that assesses tutor agents holistically using controlled student simulation and code generation tests. Extensive experiments reveal the challenges of coding tutoring and demonstrate that TRAVER achieves a significantly higher success rate. Although we use code tutoring as an example in this paper, our results and findings can be extended beyond coding, providing valuable insights into advancing tutoring agents for a variety of tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06101v2">From Conversation to Automation: Leveraging LLMs for Problem-Solving Therapy Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Problem-solving therapy (PST) is a structured psychological approach that helps individuals manage stress and resolve personal issues by guiding them through problem identification, solution brainstorming, decision-making, and outcome evaluation. As mental health care increasingly adopts technologies like chatbots and large language models (LLMs), it is important to thoroughly understand how each session of PST is conducted before attempting to automate it. We developed a comprehensive framework for PST annotation using established PST Core Strategies and a set of novel Facilitative Strategies to analyze a corpus of real-world therapy transcripts to determine which strategies are most prevalent. Using various LLMs and transformer-based models, we found that GPT-4o outperformed all models, achieving the highest accuracy (0.76) in identifying all strategies. To gain deeper insights, we examined how strategies are applied by analyzing Therapeutic Dynamics (autonomy, self-disclosure, and metaphor), and linguistic patterns within our labeled data. Our research highlights LLMs' potential to automate therapy dialogue analysis, offering a scalable tool for mental health interventions. Our framework enhances PST by improving accessibility, effectiveness, and personalized support for therapists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13259v1">HumT DumT: Measuring and controlling human-like language in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Should LLMs generate language that makes them seem human? Human-like language might improve user experience, but might also lead to overreliance and stereotyping. Assessing these potential impacts requires a systematic way to measure human-like tone in LLM outputs. We introduce HumT and SocioT, metrics for human-like tone and other dimensions of social perceptions in text data based on relative probabilities from an LLM. By measuring HumT across preference and usage datasets, we find that users prefer less human-like outputs from LLMs. HumT also offers insights into the impacts of anthropomorphism: human-like LLM outputs are highly correlated with warmth, social closeness, femininity, and low status, which are closely linked to the aforementioned harms. We introduce DumT, a method using HumT to systematically control and reduce the degree of human-like tone while preserving model performance. DumT offers a practical approach for mitigating risks associated with anthropomorphic language generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13247v1">Grounding LLM Reasoning with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) are valuable tools for representing relationships between entities in a structured format. Traditionally, these knowledge bases are queried to extract specific information. However, question-answering (QA) over such KGs poses a challenge due to the intrinsic complexity of natural language compared to the structured format and the size of these graphs. Despite these challenges, the structured nature of KGs can provide a solid foundation for grounding the outputs of Large Language Models (LLMs), offering organizations increased reliability and control. Recent advancements in LLMs have introduced reasoning methods at inference time to improve their performance and maximize their capabilities. In this work, we propose integrating these reasoning strategies with KGs to anchor every step or "thought" of the reasoning chains in KG data. Specifically, we evaluate both agentic and automated search methods across several reasoning strategies, including Chain-of-Thought (CoT), Tree-of-Thought (ToT), and Graph-of-Thought (GoT), using GRBench, a benchmark dataset for graph reasoning with domain-specific graphs. Our experiments demonstrate that this approach consistently outperforms baseline models, highlighting the benefits of grounding LLM reasoning processes in structured KG data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13233v1">SearchRAG: Can Search Engines Be Helpful for LLM-based Medical Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 8 pages, three figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in general domains but often struggle with tasks requiring specialized knowledge. Conventional Retrieval-Augmented Generation (RAG) techniques typically retrieve external information from static knowledge bases, which can be outdated or incomplete, missing fine-grained clinical details essential for accurate medical question answering. In this work, we propose SearchRAG, a novel framework that overcomes these limitations by leveraging real-time search engines. Our method employs synthetic query generation to convert complex medical questions into search-engine-friendly queries and utilizes uncertainty-based knowledge selection to filter and incorporate the most relevant and informative medical knowledge into the LLM's input. Experimental results demonstrate that our method significantly improves response accuracy in medical question answering tasks, particularly for complex questions requiring detailed and up-to-date knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13221v1">Two Tickets are Better than One: Fair and Accurate Hiring Under Strategic LLM Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      In an era of increasingly capable foundation models, job seekers are turning to generative AI tools to enhance their application materials. However, unequal access to and knowledge about generative AI tools can harm both employers and candidates by reducing the accuracy of hiring decisions and giving some candidates an unfair advantage. To address these challenges, we introduce a new variant of the strategic classification framework tailored to manipulations performed using large language models, accommodating varying levels of manipulations and stochastic outcomes. We propose a ``two-ticket'' scheme, where the hiring algorithm applies an additional manipulation to each submitted resume and considers this manipulated version together with the original submitted resume. We establish theoretical guarantees for this scheme, showing improvements for both the fairness and accuracy of hiring decisions when the true positive rate is maximized subject to a no false positives constraint. We further generalize this approach to an $n$-ticket scheme and prove that hiring outcomes converge to a fixed, group-independent decision, eliminating disparities arising from differential LLM access. Finally, we empirically validate our framework and the performance of our two-ticket scheme on real resumes using an open-source resume screening tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13189v1">MoBA: Mixture of Block Attention for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Scaling the effective context length is essential for advancing large language models (LLMs) toward artificial general intelligence (AGI). However, the quadratic increase in computational complexity inherent in traditional attention mechanisms presents a prohibitive overhead. Existing approaches either impose strongly biased structures, such as sink or window attention which are task-specific, or radically modify the attention mechanism into linear approximations, whose performance in complex reasoning tasks remains inadequately explored. In this work, we propose a solution that adheres to the ``less structure'' principle, allowing the model to determine where to attend autonomously, rather than introducing predefined biases. We introduce Mixture of Block Attention (MoBA), an innovative approach that applies the principles of Mixture of Experts (MoE) to the attention mechanism. This novel architecture demonstrates superior performance on long-context tasks while offering a key advantage: the ability to seamlessly transition between full and sparse attention, enhancing efficiency without the risk of compromising performance. MoBA has already been deployed to support Kimi's long-context requests and demonstrates significant advancements in efficient attention computation for LLMs. Our code is available at https://github.com/MoonshotAI/MoBA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v1">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13125v1">RuozhiBench: Evaluating LLMs with Logical Fallacies and Misleading Premises</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have shown that they can answer questions requiring complex reasoning. However, their ability to identify and respond to text containing logical fallacies or deliberately misleading premises remains less studied. To address this gap, we introduce RuozhiBench, a bilingual dataset comprising 677 carefully curated questions that contain various forms of deceptive reasoning, meticulously crafted through extensive human effort and expert review. In a comprehensive evaluation of 17 LLMs from 5 Series over RuozhiBench using both open-ended and two-choice formats, we conduct extensive analyses on evaluation protocols and result patterns. Despite their high scores on conventional benchmarks, these models showed limited ability to detect and reason correctly about logical fallacies, with even the best-performing model, Claude-3-haiku, achieving only 62% accuracy compared to the human of more than 90%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13120v1">Adapting Psycholinguistic Research for LLMs: Gender-inclusive Language in a Coreference Context</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 9 pages, 7 figures, submitted to ACL 2025 (ARR February 2025 cycle)
    </div>
    <details class="paper-abstract">
      Gender-inclusive language is often used with the aim of ensuring that all individuals, regardless of gender, can be associated with certain concepts. While psycholinguistic studies have examined its effects in relation to human cognition, it remains unclear how Large Language Models (LLMs) process gender-inclusive language. Given that commercial LLMs are gaining an increasingly strong foothold in everyday applications, it is crucial to examine whether LLMs in fact interpret gender-inclusive language neutrally, because the language they generate has the potential to influence the language of their users. This study examines whether LLM-generated coreferent terms align with a given gender expression or reflect model biases. Adapting psycholinguistic methods from French to English and German, we find that in English, LLMs generally maintain the antecedent's gender but exhibit underlying masculine bias. In German, this bias is much stronger, overriding all tested gender-neutralization strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13107v1">MatterChat: A Multi-Modal LLM for Material Science</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Understanding and predicting the properties of inorganic materials is crucial for accelerating advancements in materials science and driving applications in energy, electronics, and beyond. Integrating material structure data with language-based information through multi-modal large language models (LLMs) offers great potential to support these efforts by enhancing human-AI interaction. However, a key challenge lies in integrating atomic structures at full resolution into LLMs. In this work, we introduce MatterChat, a versatile structure-aware multi-modal LLM that unifies material structural data and textual inputs into a single cohesive model. MatterChat employs a bridging module to effectively align a pretrained machine learning interatomic potential with a pretrained LLM, reducing training costs and enhancing flexibility. Our results demonstrate that MatterChat significantly improves performance in material property prediction and human-AI interaction, surpassing general-purpose LLMs such as GPT-4. We also demonstrate its usefulness in applications such as more advanced scientific reasoning and step-by-step material synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01077v2">Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted outputs, posing a serious threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This disrupts the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the "unsafe" prediction rate, bypassing existing safeguards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15939v2">CausalGraph2LLM: Evaluating LLMs for Causal Queries</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 NAACL'25 Findings, Code - https://github.com/ivaxi0s/CausalGraph2LLM
    </div>
    <details class="paper-abstract">
      Causality is essential in scientific research, enabling researchers to interpret true relationships between variables. These causal relationships are often represented by causal graphs, which are directed acyclic graphs. With the recent advancements in Large Language Models (LLMs), there is an increasing interest in exploring their capabilities in causal reasoning and their potential use to hypothesize causal graphs. These tasks necessitate the LLMs to encode the causal graph effectively for subsequent downstream tasks. In this paper, we introduce CausalGraph2LLM, a comprehensive benchmark comprising over 700k queries across diverse causal graph settings to evaluate the causal reasoning capabilities of LLMs. We categorize the causal queries into two types: graph-level and node-level queries. We benchmark both open-sourced and propriety models for our study. Our findings reveal that while LLMs show promise in this domain, they are highly sensitive to the encoding used. Even capable models like GPT-4 and Gemini-1.5 exhibit sensitivity to encoding, with deviations of about $60\%$. We further demonstrate this sensitivity for downstream causal intervention tasks. Moreover, we observe that LLMs can often display biases when presented with contextual information about a causal graph, potentially stemming from their parametric memory.
    </details>
</div>
