# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18125v1">HyperG: Hypergraph-Enhanced LLMs for Structured Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Given that substantial amounts of domain-specific knowledge are stored in structured formats, such as web data organized through HTML, Large Language Models (LLMs) are expected to fully comprehend this structured information to broaden their applications in various real-world downstream tasks. Current approaches for applying LLMs to structured data fall into two main categories: serialization-based and operation-based methods. Both approaches, whether relying on serialization or using SQL-like operations as an intermediary, encounter difficulties in fully capturing structural relationships and effectively handling sparse data. To address these unique characteristics of structured data, we propose HyperG, a hypergraph-based generation framework aimed at enhancing LLMs' ability to process structured knowledge. Specifically, HyperG first augment sparse data with contextual information, leveraging the generative power of LLMs, and incorporate a prompt-attentive hypergraph learning (PHL) network to encode both the augmented information and the intricate structural relationships within the data. To validate the effectiveness and generalization of HyperG, we conduct extensive experiments across two different downstream tasks requiring structured knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18116v1">Bayesian Optimization for Controlled Image Editing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 8 figures
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18080v1">Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v2">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18697v2">How Good Are LLMs for Literary Translation, Really? Literary Translation Evaluation with Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 NAACL Camera-Ready version
    </div>
    <details class="paper-abstract">
      Recent research has focused on literary machine translation (MT) as a new challenge in MT. However, the evaluation of literary MT remains an open problem. We contribute to this ongoing discussion by introducing LITEVAL-CORPUS, a paragraph-level parallel corpus containing verified human translations and outputs from 9 MT systems, which totals over 2k translations and 13k evaluated sentences across four language pairs, costing 4.5k C. This corpus enables us to (i) examine the consistency and adequacy of human evaluation schemes with various degrees of complexity, (ii) compare evaluations by students and professionals, assess the effectiveness of (iii) LLM-based metrics and (iv) LLMs themselves. Our findings indicate that the adequacy of human evaluation is controlled by two factors: the complexity of the evaluation scheme (more complex is less adequate) and the expertise of evaluators (higher expertise yields more adequate evaluations). For instance, MQM (Multidimensional Quality Metrics), a complex scheme and the de facto standard for non-literary human MT evaluation, is largely inadequate for literary translation evaluation: with student evaluators, nearly 60% of human translations are misjudged as indistinguishable or inferior to machine translations. In contrast, BWS (BEST-WORST SCALING), a much simpler scheme, identifies human translations at a rate of 80-100%. Automatic metrics fare dramatically worse, with rates of at most 20%. Our overall evaluation indicates that published human translations consistently outperform LLM translations, where even the most recent LLMs tend to produce considerably more literal and less diverse translations compared to humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18036v1">Harnessing Multiple Large Language Models: A Survey on LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages, 2 figures, codebase: https://github.com/junchenzhi/Awesome-LLM-Ensemble
    </div>
    <details class="paper-abstract">
      LLM Ensemble -- which involves the comprehensive use of multiple large language models (LLMs), each aimed at handling user queries during downstream inference, to benefit from their individual strengths -- has gained substantial attention recently. The widespread availability of LLMs, coupled with their varying strengths and out-of-the-box usability, has profoundly advanced the field of LLM Ensemble. This paper presents the first systematic review of recent developments in LLM Ensemble. First, we introduce our taxonomy of LLM Ensemble and discuss several related research problems. Then, we provide a more in-depth classification of the methods under the broad categories of "ensemble-before-inference, ensemble-during-inference, ensemble-after-inference", and review all relevant methods. Finally, we introduce related benchmarks and applications, summarize existing studies, and suggest several future research directions. A curated list of papers on LLM Ensemble is available at https://github.com/junchenzhi/Awesome-LLM-Ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13178v3">SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 23 pages, 17 tables, 8 figures
    </div>
    <details class="paper-abstract">
      With the integration of large language models (LLMs), embodied agents have strong capabilities to process the scene information and plan complicated instructions in natural language, paving the way for the potential deployment of embodied robots. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. To study this issue, we present SafeAgentBench-a new benchmark for safety-aware task planning of embodied LLM agents. SafeAgentBench includes: (1) a new dataset with 750 tasks, covering 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 8 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10\% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. More details and code are available at https://github.com/shengyin1224/SafeAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19487v4">HealthQ: Unveiling Questioning Capabilities of LLM Chains in Healthcare Conversations</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Effective patient care in digital healthcare requires large language models (LLMs) that not only answer questions but also actively gather critical information through well-crafted inquiries. This paper introduces HealthQ, a novel framework for evaluating the questioning capabilities of LLM healthcare chains. By implementing advanced LLM chains, including Retrieval-Augmented Generation (RAG), Chain of Thought (CoT), and reflective chains, HealthQ assesses how effectively these chains elicit comprehensive and relevant patient information. To achieve this, we integrate an LLM judge to evaluate generated questions across metrics such as specificity, relevance, and usefulness, while aligning these evaluations with traditional Natural Language Processing (NLP) metrics like ROUGE and Named Entity Recognition (NER)-based set comparisons. We validate HealthQ using two custom datasets constructed from public medical datasets, ChatDoctor and MTS-Dialog, and demonstrate its robustness across multiple LLM judge models, including GPT-3.5, GPT-4, and Claude. Our contributions are threefold: we present the first systematic framework for assessing questioning capabilities in healthcare conversations, establish a model-agnostic evaluation methodology, and provide empirical evidence linking high-quality questions to improved patient information elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02487v3">LiCoEval: Evaluating LLMs on License Compliance in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 The 47th International Conference on Software Engineering(ICSE 2025)
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have revolutionized code generation, leading to widespread adoption of AI coding tools by developers. However, LLMs can generate license-protected code without providing the necessary license information, leading to potential intellectual property violations during software production. This paper addresses the critical, yet underexplored, issue of license compliance in LLM-generated code by establishing a benchmark to evaluate the ability of LLMs to provide accurate license information for their generated code. To establish this benchmark, we conduct an empirical study to identify a reasonable standard for "striking similarity" that excludes the possibility of independent creation, indicating a copy relationship between the LLM output and certain open-source code. Based on this standard, we propose LiCoEval, to evaluate the license compliance capabilities of LLMs, i.e., the ability to provide accurate license or copyright information when they generate code with striking similarity to already existing copyrighted code. Using LiCoEval, we evaluate 14 popular LLMs, finding that even top-performing LLMs produce a non-negligible proportion (0.88% to 2.01%) of code strikingly similar to existing open-source implementations. Notably, most LLMs fail to provide accurate license information, particularly for code under copyleft licenses. These findings underscore the urgent need to enhance LLM compliance capabilities in code generation tasks. Our study provides a foundation for future research and development to improve license compliance in AI-assisted software development, contributing to both the protection of open-source software copyrights and the mitigation of legal risks for LLM users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17967v1">LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at https://github.com/wekjsdvnm/Agent-Trading-Arena.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v3">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17898v1">VeriPlan: Integrating Formal Verification and LLMs into End-User Planning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 In CHI Conference on Human Factors in Computing Systems (CHI '25), April 26-May 1, 2025, Yokohama, Japan. ACM, New York, NY, USA, 19 pages
    </div>
    <details class="paper-abstract">
      Automated planning is traditionally the domain of experts, utilized in fields like manufacturing and healthcare with the aid of expert planning tools. Recent advancements in LLMs have made planning more accessible to everyday users due to their potential to assist users with complex planning tasks. However, LLMs face several application challenges within end-user planning, including consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a system that applies formal verification techniques, specifically model checking, to enhance the reliability and flexibility of LLMs for end-user planning. In addition to the LLM planner, VeriPlan includes three additional core features -- a rule translator, flexibility sliders, and a model checker -- that engage users in the verification process. Through a user study (n=12), we evaluate VeriPlan, demonstrating improvements in the perceived quality, usability, and user satisfaction of LLMs. Our work shows the effective integration of formal verification and user-control features with LLMs for end-user planning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17882v1">Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at https://hankleid.github.io/ProjectMundo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17878v1">Towards Enhanced Immersion and Agency for LLM-based Interactive Drama</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      LLM-based Interactive Drama is a novel AI-based dialogue scenario, where the user (i.e. the player) plays the role of a character in the story, has conversations with characters played by LLM agents, and experiences an unfolding story. This paper begins with understanding interactive drama from two aspects: Immersion, the player's feeling of being present in the story, and Agency, the player's ability to influence the story world. Both are crucial to creating an enjoyable interactive experience, while they have been underexplored in previous work. To enhance these two aspects, we first propose Playwriting-guided Generation, a novel method that helps LLMs craft dramatic stories with substantially improved structures and narrative quality. Additionally, we introduce Plot-based Reflection for LLM agents to refine their reactions to align with the player's intentions. Our evaluation relies on human judgment to assess the gains of our methods in terms of immersion and agency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17857v1">SYNTHEMPATHY: A Scalable Empathy Corpus Generated Using LLMs Without Any Crowdsourcing</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Previous research has shown that humans are more receptive towards language models that that exhibit empathetic behavior. While empathy is essential for developing helpful dialogue agents, very few large corpora containing empathetic dialogues are available for fine-tune LLMs. The few existing corpora have largely relied on crowdsourcing to simulate empathetic conversations, a process that is expensive, time-consuming, and not scalable to larger datasets. We propose a data generation framework for developing SYNTHEMPATHY, a large corpus containing 105k empathetic responses to real-life situations compiled through LLM generation. A base Mistral 7B model fine-tuned on our SYNTHEMPATHY corpus exhibits an increase in the average empathy score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06364v2">Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Adapting pre-trained large language models (LLMs) is crucial but challenging due to their enormous size. Parameter-efficient fine-tuning (PEFT) techniques typically employ additive adapters applied to frozen model weights. To further reduce memory usage, model weights can be compressed through quantization. However, existing PEFT methods often yield suboptimal model quality due to restrictive assumptions, such as imposing low-rank constraints on adapters to reduce trainable parameters. We find that sketching, a popular data compression technique, can serve as an efficient adaptation strategy for LLMs while avoiding low-rank assumptions. We introduce SketchTune, a compressive adaptation strategy that compresses LLM weights into compact fine-tunable sketches, integrating compression and adaptation into a unified framework. This integration eliminates the need for complex two-path computation common in existing PEFT techniques, enabling faster and more memory-efficient training and inference. SketchTune is supported by mathematical insights into matrix classes that are better approximated using sketching rather than low-rank methods. Our rigorous evaluations with Llama-1/2/3 models demonstrate that SketchTune outperforms leading PEFT methods across diverse tasks including math problem-solving, common sense reasoning, and instruction following, while using substantially smaller base models and comparable trainable parameters. As a highlight, SketchTune outperforms LoRA, DoRA, and S2FT on commonsense and math benchmarks using 2.6-3.5$\times$ smaller base models and exceeds LoftQ in accuracy by 14.48% on GSM8K with 7.3$\times$ fewer trainable parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17823v1">A General Framework to Enhance Fine-tuning-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      Unlearning has been proposed to remove copyrighted and privacy-sensitive data from Large Language Models (LLMs). Existing approaches primarily rely on fine-tuning-based methods, which can be categorized into gradient ascent-based (GA-based) and suppression-based methods. However, they often degrade model utility (the ability to respond to normal prompts). In this work, we aim to develop a general framework that enhances the utility of fine-tuning-based unlearning methods. To achieve this goal, we first investigate the common property between GA-based and suppression-based methods. We unveil that GA-based methods unlearn by distinguishing the target data (i.e., the data to be removed) and suppressing related generations, which is essentially the same strategy employed by suppression-based methods. Inspired by this finding, we introduce Gated Representation UNlearning (GRUN) which has two components: a soft gate function for distinguishing target data and a suppression module using Representation Fine-tuning (ReFT) to adjust representations rather than model parameters. Experiments show that GRUN significantly improves the unlearning and utility. Meanwhile, it is general for fine-tuning-based methods, efficient and promising for sequential unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09179v2">Towards Effective Evaluations and Comparisons for LLM Unlearning Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-25
    </div>
    <details class="paper-abstract">
      The imperative to eliminate undesirable data memorization underscores the significance of machine unlearning for large language models (LLMs). Recent research has introduced a series of promising unlearning methods, notably boosting the practical significance of the field. Nevertheless, adopting a proper evaluation framework to reflect the true unlearning efficacy is also essential yet has not received adequate attention. This paper seeks to refine the evaluation of LLM unlearning by addressing two key challenges -- a) the robustness of evaluation metrics and b) the trade-offs between competing goals. The first challenge stems from findings that current metrics are susceptible to various red teaming scenarios. It indicates that they may not reflect the true extent of knowledge retained by LLMs but rather tend to mirror superficial model behaviors, thus prone to attacks. We address this issue by devising and assessing a series of candidate metrics, selecting the most robust ones under various types of attacks. The second challenge arises from the conflicting goals of eliminating unwanted knowledge while retaining those of others. This trade-off between unlearning and retention often fails to conform the Pareto frontier, rendering it subtle to compare the efficacy between methods that excel only in either unlearning or retention. We handle this issue by proposing a calibration method that can restore the original performance on non-targeted data after unlearning, thereby allowing us to focus exclusively on assessing the strength of unlearning. Our evaluation framework notably enhances the effectiveness when assessing and comparing various LLM unlearning methods, further allowing us to benchmark existing works, identify their proper hyper-parameters, and explore new tricks to enhance their practical efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v3">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 36 pages. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17812v1">Can Multimodal LLMs Perform Time Series Anomaly Detection?</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 9 pages for the main content; 32 pages for the full paper including the appendix. More resources on the intersection of multimodal LLMs and time series analysis are on the website https://mllm-ts.github.io
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly used in time series analysis. However, the potential of multimodal LLMs (MLLMs), particularly vision-language models, for time series remains largely under-explored. One natural way for humans to detect time series anomalies is through visualization and textual description. Motivated by this, we raise a critical and practical research question: Can multimodal LLMs perform time series anomaly detection? To answer this, we propose VisualTimeAnomaly benchmark to evaluate MLLMs in time series anomaly detection (TSAD). Our approach transforms time series numerical data into the image format and feed these images into various MLLMs, including proprietary models (GPT-4o and Gemini-1.5) and open-source models (LLaVA-NeXT and Qwen2-VL), each with one larger and one smaller variant. In total, VisualTimeAnomaly contains 12.4k time series images spanning 3 scenarios and 3 anomaly granularities with 9 anomaly types across 8 MLLMs. Starting with the univariate case (point- and range-wise anomalies), we extend our evaluation to more practical scenarios, including multivariate and irregular time series scenarios, and variate-wise anomalies. Our study reveals several key insights: 1) MLLMs detect range- and variate-wise anomalies more effectively than point-wise anomalies. 2) MLLMs are highly robust to irregular time series, even with 25% of the data missing. 3) Open-source MLLMs perform comparably to proprietary models in TSAD. While open-source MLLMs excel on univariate time series, proprietary MLLMs demonstrate superior effectiveness on multivariate time series. To the best of our knowledge, this is the first work to comprehensively investigate MLLMs for TSAD, particularly for multivariate and irregular time series scenarios. We release our dataset and code at https://github.com/mllm-ts/VisualTimeAnomaly to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10573v2">Putting People in LLMs' Shoes: Generating Better Answers via Question Rewriter</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 7 pages, 4 figures, 5 tables and accepted at AAAI 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant capabilities, particularly in the domain of question answering (QA). However, their effectiveness in QA is often undermined by the vagueness of user questions. To address this issue, we introduce single-round instance-level prompt optimization, referred to as question rewriter. By enhancing the intelligibility of human questions for black-box LLMs, our question rewriter improves the quality of generated answers. The rewriter is optimized using direct preference optimization based on feedback collected from automatic criteria for evaluating generated answers; therefore, its training does not require costly human annotations. The experiments across multiple black-box LLMs and long-form question answering (LFQA) datasets demonstrate the efficacy of our method. This paper provides a practical framework for training question rewriters and sets a precedent for future explorations in prompt optimization within LFQA tasks. Code is available at https://github.com/3244we/Question-Rewriter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.04482v1">Automatic Bug Detection in LLM-Powered Text-Based Games Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-25
      | 💬 Accepted for publication in Findings of the Association for Computational Linguistics: ACL 2024
    </div>
    <details class="paper-abstract">
      Advancements in large language models (LLMs) are revolutionizing interactive game design, enabling dynamic plotlines and interactions between players and non-player characters (NPCs). However, LLMs may exhibit flaws such as hallucinations, forgetfulness, or misinterpretations of prompts, causing logical inconsistencies and unexpected deviations from intended designs. Automated techniques for detecting such game bugs are still lacking. To address this, we propose a systematic LLM-based method for automatically identifying such bugs from player game logs, eliminating the need for collecting additional data such as post-play surveys. Applied to a text-based game DejaBoom!, our approach effectively identifies bugs inherent in LLM-powered interactive games, surpassing unstructured LLM-powered bug-catching methods and filling the gap in automated detection of logical and design flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16759v1">The Blessing of Reasoning: LLM-Based Contrastive Explanations in Black-Box Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Modern recommender systems use ML models to predict consumer preferences from consumption history. Although these "black-box" models achieve impressive predictive performance, they often suffer from a lack of transparency and explainability. Contrary to the presumed tradeoff between explainability and accuracy, we show that integrating large language models (LLMs) with deep neural networks (DNNs) can improve both. We propose LR-Recsys, which augments DNN-based systems with LLM reasoning capabilities. LR-Recsys introduces a contrastive-explanation generator that produces human-readable positive explanations and negative explanations. These explanations are embedded via a fine-tuned autoencoder and combined with consumer and product features to improve predictions. Beyond offering explainability, we show that LR-Recsys also improves learning efficiency and predictive accuracy, as supported by high-dimensional, multi-environment statistical learning theory. LR-Recsys outperforms state-of-the-art recommender systems by 3-14% on three real-world datasets. Importantly, our analysis reveals that these gains primarily derive from LLMs' reasoning capabilities rather than their external domain knowledge. LR-RecSys presents an effective approach to combine LLMs with traditional DNNs, two of the most widely used ML models today. The explanations generated by LR-Recsys provide actionable insights for consumers, sellers, and platforms, helping to build trust, optimize product offerings, and inform targeting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17728v1">LLM Inference Acceleration via Efficient Operation Fusion</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The rapid development of the Transformer-based Large Language Models (LLMs) in recent years has been closely linked to their ever-growing and already enormous sizes. Many LLMs contain hundreds of billions of parameters and require dedicated hardware resources for training and inference. One of the key challenges inherent to the Transformer architecture is the requirement to support numerous non-linear transformations that involves normalization. For instance, each decoder block typically contains at least one Softmax operation and two Layernorms. The computation of the corresponding normalization scaling factors becomes a major bottleneck as it requires spatial collective operations. In other words, when it comes to the computation of denominators for Softmax and Layernorm, all vector elements must be aggregated into a single location, requiring significant communication. These collective operations slow down inference on Transformers by approximately 20%, defeating the whole purpose of distributed in-memory compute. In this work, we propose an extremely efficient technique that can completely hide the overhead caused by such collective operations. Note that each Softmax and Layernorm operation is typically followed by a linear layer. Since non-linear and linear operations are performed on different hardware engines, they can be easily parallelized once the algebra allows such commutation. By leveraging the inherent properties of linear operations, we can defer the normalization of the preceding Softmax and Layernorm until after the linear layer is computed. Now we can compute the collective scaling factors concurrently with the matrix multiplication and completely hide the latency of the former behind the latter. Such parallelization preserves the numerical accuracy while significantly improving the hardware utilization and reducing the overall latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17701v1">From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 24 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Evacuation decision prediction is critical for efficient and effective wildfire response by helping emergency management anticipate traffic congestion and bottlenecks, allocate resources, and minimize negative impacts. Traditional statistical methods for evacuation decision prediction fail to capture the complex and diverse behavioral logic of different individuals. In this work, for the first time, we introduce FLARE, short for facilitating LLM for advanced reasoning on wildfire evacuation decision prediction, a Large Language Model (LLM)-based framework that integrates behavioral theories and models to streamline the Chain-of-Thought (CoT) reasoning and subsequently integrate with memory-based Reinforcement Learning (RL) module to provide accurate evacuation decision prediction and understanding. Our proposed method addresses the limitations of using existing LLMs for evacuation behavioral predictions, such as limited survey data, mismatching with behavioral theory, conflicting individual preferences, implicit and complex mental states, and intractable mental state-behavior mapping. Experiments on three post-wildfire survey datasets show an average of 20.47% performance improvement over traditional theory-informed behavioral models, with strong cross-event generalizability. Our complete code is publicly available at https://github.com/SusuXu-s-Lab/FLARE
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00151v4">Scheherazade: Evaluating Chain-of-Thought Math Reasoning in LLMs with Chain-of-Problems</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Benchmarks are critical for measuring Large Language Model (LLM) reasoning capabilities. Some benchmarks have even become the de facto indicator of such capabilities. However, as LLM reasoning capabilities improve, existing widely-used benchmarks such as GSM8K marginally encapsulate model reasoning differentials - most state-of-the-art models for example achieve over 94% accuracy on the GSM8K dataset (paperwithcode, 2024). While constructing harder benchmarks is possible, their creation is often manual, expensive, and unscalable. As such, we present Scheherazade, an automated approach to produce large quantities of challenging mathematical reasoning benchmarks by logically chaining a small starting set of problems. We propose two different chaining methods, forward chaining and backward chaining, which include randomized branching techniques to generate complex reasoning problems. We apply Scheherazade on GSM8K to create GSM8K-Scheherazade and evaluate 3 frontier LLMs and OpenAI's o1-preview on it. We show that while other frontier models' performance declines precipitously at only a few questions chained, our evaluation suggests o1-preview's performance persists, with the flagship OpenAI model the only one to perform better at backward reasoning. Our data and code are available at https://github.com/YoshikiTakashima/scheherazade-code-data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17187v2">Visualizing Uncertainty in Translation Tasks: An Evaluation of LLM Performance and Confidence Metrics</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 We would like to withdraw our paper due to an error in the experimental methodology, which impacts the validity of our results. The error specifically affects the analysis presented in the Discussion, where an incorrect experimental modeling step led to misleading interpretations
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized for machine translation, yet their predictions often exhibit uncertainties that hinder interpretability and user trust. Effectively visualizing these uncertainties can enhance the usability of LLM outputs, particularly in contexts where translation accuracy is critical. This paper addresses two primary objectives: (1) providing users with token-level insights into model confidence and (2) developing a web-based visualization tool to quantify and represent translation uncertainties. To achieve these goals, we utilized the T5 model with the WMT19 dataset for translation tasks and evaluated translation quality using established metrics such as BLEU, METEOR, and ROUGE. We introduced three novel uncertainty quantification (UQ) metrics: (1) the geometric mean of token probabilities, (2) the arithmetic mean of token probabilities, and (3) the arithmetic mean of the kurtosis of token distributions. These metrics provide a simple yet effective framework for evaluating translation performance. Our analysis revealed a linear relationship between the traditional evaluation metrics and our UQ metrics, demonstrating the validity of our approach. Additionally, we developed an interactive web-based visualization that uses a color gradient to represent token confidence. This tool offers users a clear and intuitive understanding of translation quality while providing valuable insights into model performance. Overall, we show that our UQ metrics and visualization are both robust and interpretable, offering practical tools for evaluating and accessing machine translation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12299v2">Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance across many tasks, yet aligning them with desired behaviors remains challenging. Activation intervention has emerged as an effective and economical method to modify the behavior of LLMs. Despite considerable interest in this area, current intervention methods exclusively employ a fixed steering vector to modify model activations, lacking adaptability to diverse input semantics. To address this limitation, we propose Semantics-Adaptive Dynamic Intervention (SADI), a novel method that constructs a dynamic steering vector to intervene model activations at inference time. More specifically, SADI utilizes activation differences in contrastive pairs to precisely identify critical elements of an LLM (i.e., attention heads, hidden states, and neurons) for targeted intervention. During inference, SADI dynamically steers model behavior by scaling element-wise activations based on the directions of input semantics. Experimental results show that SADI outperforms established baselines by substantial margins, improving task performance without training. SADI's cost-effectiveness and generalizability across various LLM backbones and tasks highlight its potential as a versatile alignment technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17650v1">Wearable Meets LLM for Stress Management: A Duoethnographic Study Integrating Wearable-Triggered Stressors and LLM Chatbots for Personalized Interventions</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 In CHI '25 Proceedings of the CHI Conference on Human Factors in Computing Systems Yokohama, Japan
    </div>
    <details class="paper-abstract">
      We use a duoethnographic approach to study how wearable-integrated LLM chatbots can assist with personalized stress management, addressing the growing need for immediacy and tailored interventions. Two researchers interacted with custom chatbots over 22 days, responding to wearable-detected physiological prompts, recording stressor phrases, and using them to seek tailored interventions from their LLM-powered chatbots. They recorded their experiences in autoethnographic diaries and analyzed them during weekly discussions, focusing on the relevance, clarity, and impact of chatbot-generated interventions. Results showed that even though most events triggered by the wearable were meaningful, only one in five warranted an intervention. It also showed that interventions tailored with brief event descriptions were more effective than generic ones. By examining the intersection of wearables and LLM, this research contributes to developing more effective, user-centric mental health tools for real-time stress relief and behavior change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17638v1">Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Legal services rely heavily on text processing. While large language models (LLMs) show promise, their application in legal contexts demands higher accuracy, repeatability, and transparency. Logic programs, by encoding legal concepts as structured rules and facts, offer reliable automation, but require sophisticated text extraction. We propose a neuro-symbolic approach that integrates LLMs' natural language understanding with logic-based reasoning to address these limitations. As a legal document case study, we applied neuro-symbolic AI to coverage-related queries in insurance contracts using both closed and open-source LLMs. While LLMs have improved in legal reasoning, they still lack the accuracy and consistency required for complex contract analysis. In our analysis, we tested three methodologies to evaluate whether a specific claim is covered under a contract: a vanilla LLM, an unguided approach that leverages LLMs to encode both the contract and the claim, and a guided approach that uses a framework for the LLM to encode the contract. We demonstrated the promising capabilities of LLM + Logic in the guided approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13347v2">Craw4LLM: Efficient Web Crawling for LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Web crawl is a main source of large language models' (LLMs) pretraining data, but the majority of crawled web pages are discarded in pretraining due to low data quality. This paper presents Craw4LLM, an efficient web crawling method that explores the web graph based on the preference of LLM pretraining. Specifically, it leverages the influence of a webpage in LLM pretraining as the priority score of the web crawler's scheduler, replacing the standard graph connectivity based priority. Our experiments on a web graph containing 900 million webpages from a commercial search engine's index demonstrate the efficiency of Craw4LLM in obtaining high-quality pretraining data. With just 21% URLs crawled, LLMs pretrained on Craw4LLM data reach the same downstream performances of previous crawls, significantly reducing the crawling waste and alleviating the burdens on websites. Our code is publicly available at https://github.com/cxcscmu/Craw4LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17606v1">ELMo-Tune-V2: LLM-Assisted Full-Cycle Auto-Tuning to Optimize LSM-Based Key-Value Stores</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Log-Structured Merge-tree-based Key-Value Store (LSM-KVS) is a foundational storage engine serving diverse modern workloads, systems, and applications. To suit varying use cases, LSM-KVS allows a vast configuration space that controls core parameters like compaction, flush, and cache sizes, each consuming a shared pool of CPU, Memory, and Storage resources. Navigating the LSM-KVS configuration space necessitates knowledge of the impact of each configuration on the expected workload and underlying hardware. Beyond expensive and time-intensive human-expert-based tuning, existing LSM-KVS tuning solutions focus on tuning with specific workload expectations while limited to a narrow subset of parameters. This paper introduces ELMo-Tune-V2, a framework that integrates Large Language Models (LLMs) at its foundation to demonstrate the potential of applying modern LLMs in data system optimization problems. ELMo-Tune-V2 leverages the contextual reasoning, cross-domain, and generative capabilities of LLMs to perform 1) self-navigated characterization and modeling of LSM-KVS workloads, 2) automatic tuning across a broad parameter space using cross-domain knowledge, and 3) real-time dynamic configuration adjustments for LSM-KVS. ELMo-Tune-V2 integrates three innovations: LLM-based workload synthesis for adaptive benchmark generation, feedback-driven iterative fine-tuning for configuration refinement, and real-time tuning to handle evolving workloads. Through detailed evaluation using RocksDB under several real-world applications across diverse scenarios, ELMo-Tune-V2 achieves performance improvements up to ~14X our YCSB benchmarks compared against default RocksDB configurations, and our end-to-end tests with upper-level applications, NebulaGraph and Kvrocks, demonstrate performance gains of 34% and 26%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17598v1">Hallucination Detection in LLMs Using Spectral Features of Attention Maps</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14145v2">LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 In submission to INTERSPEECH 2025
    </div>
    <details class="paper-abstract">
      Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v2">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various natural language tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs. This paper summarizes and categorizes the main challenges into two aspects: (1) Logical question answering, LLMs often fail to generate the correct answer within complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency, LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art Macaw question-answering LLM answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose detailed taxonomies of these methods. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, pretraining, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistency, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extensions to modal logic to account for uncertainty, and efficient algorithms satisfying multiple logical consistencies simultaneously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17535v1">The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17529v1">ConvoyLLM: Dynamic Multi-Lane Convoy Control Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      This paper proposes a novel method for multi-lane convoy formation control that uses large language models (LLMs) to tackle coordination challenges in dynamic highway environments. Each connected and autonomous vehicle in the convoy uses a knowledge-driven approach to make real-time adaptive decisions based on various scenarios. Our method enables vehicles to dynamically perform tasks, including obstacle avoidance, convoy joining/leaving, and escort formation switching, all while maintaining the overall convoy structure. We design a Interlaced formation control strategy based on locally dynamic distributed graphs, ensuring the convoy remains stable and flexible. We conduct extensive experiments in the SUMO simulation platform across multiple traffic scenarios, and the results demonstrate that the proposed method is effective, robust, and adaptable to dynamic environments. The code is available at: https://github.com/chuduanfeng/ConvoyLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18528v1">ARACNE: An LLM-Based Autonomous Shell Pentesting Agent</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 7 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      We introduce ARACNE, a fully autonomous LLM-based pentesting agent tailored for SSH services that can execute commands on real Linux shell systems. Introduces a new agent architecture with multi-LLM model support. Experiments show that ARACNE can reach a 60\% success rate against the autonomous defender ShelLM and a 57.58\% success rate against the Over The Wire Bandit CTF challenges, improving over the state-of-the-art. When winning, the average number of actions taken by the agent to accomplish the goals was less than 5. The results show that the use of multi-LLM is a promising approach to increase accuracy in the actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v1">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 10 pages, 9 figures
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding: it asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v1">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19381v4">HYBRIDMIND: Meta Selection of Natural Language and Symbolic Language for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      LLMs approach logical and mathematical reasoning through natural or symbolic languages. While natural language offers human-accessible flexibility but suffers from ambiguity, symbolic reasoning provides precise, machine-executable inferences at the cost of strict domain constraints. We introduce HYBRIDMIND, an adaptive strategy that selects the optimal reasoning approach for each reasoning problem. Through extensive experiments, we evaluate both prompting-based approaches with state-of-the-art LLMs and fine-tuned open-source models. We find that fine-tuning LLaMA-3.1-8B-Instruct as a meta-selector outperforms GPT-4o's natural language reasoning by 4.4\% on FOLIO and 1.3\% on MATH. More notably, using GPT-3.5-turbo as a prompted meta-selector yields a 10\% improvement on FOLIO's challenging subset compared to GPT-4o. We will release our code and data to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05315v3">Aligned at the Start: Conceptual Groupings in LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      This paper shifts focus to the often-overlooked input embeddings - the initial representations fed into transformer blocks. Using fuzzy graph, k-nearest neighbor (k-NN), and community detection, we analyze embeddings from diverse LLMs, finding significant categorical community structure aligned with predefined concepts and categories aligned with humans. We observe these groupings exhibit within-cluster organization (such as hierarchies, topological ordering, etc.), hypothesizing a fundamental structure that precedes contextual processing. To further investigate the conceptual nature of these groupings, we explore cross-model alignments across different LLM categories within their input embeddings, observing a medium to high degree of alignment. Furthermore, provide evidence that manipulating these groupings can play a functional role in mitigating ethnicity bias in LLM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06287v2">Non-Halting Queries: Exploiting Fixed Points in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      We introduce a new vulnerability that exploits fixed points in autoregressive models and use it to craft queries that never halt. More precisely, for non-halting queries, the LLM never samples the end-of-string token <eos>. We rigorously analyze the conditions under which the non-halting anomaly presents itself. In particular, at temperature zero, we prove that if a repeating (cyclic) token sequence is observed at the output beyond the context size, then the LLM does not halt. We demonstrate non-halting queries in many experiments performed in base unaligned models where repeating prompts immediately lead to a non-halting cyclic behavior as predicted by the analysis. Further, we develop a simple recipe that takes the same fixed points observed in the base model and creates a prompt structure to target aligned models. We demonstrate the recipe's success in sending every major model released over the past year into a non-halting state with the same simple prompt even over higher temperatures. Further, we devise an experiment with 100 randomly selected tokens and show that the recipe to create non-halting queries succeeds with high success rates ranging from 97% for GPT-4o to 19% for Gemini Pro 1.5. These results show that the proposed adversarial recipe succeeds in bypassing alignment at one to two orders of magnitude higher rates compared to earlier reports. We also study gradient-based direct inversion using ARCA to craft new short prompts to induce the non-halting state. We inverted 10,000 random repeating 2-cycle outputs for llama-3.1-8b-instruct. Out of 10,000 three-token inverted prompts 1,512 yield non-halting queries reaching a rate of 15%. Our experiments with ARCA show that non-halting may be easily induced with as few as 3 input tokens with high probability. Overall, our experiments demonstrate that non-halting queries are prevalent and relatively easy to find.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17341v1">Time series forecasting based on optimized LLM for fault prediction in distribution power grid insulators</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Surface contamination on electrical grid insulators leads to an increase in leakage current until an electrical discharge occurs, which can result in a power system shutdown. To mitigate the possibility of disruptive faults resulting in a power outage, monitoring contamination and leakage current can help predict the progression of faults. Given this need, this paper proposes a hybrid deep learning (DL) model for predicting the increase in leakage current in high-voltage insulators. The hybrid structure considers a multi-criteria optimization using tree-structured Parzen estimation, an input stage filter for signal noise attenuation combined with a large language model (LLM) applied for time series forecasting. The proposed optimized LLM outperforms state-of-the-art DL models with a root-mean-square error equal to 2.24$\times10^{-4}$ for a short-term horizon and 1.21$\times10^{-3}$ for a medium-term horizon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17328v1">Mutual Reinforcement of LLM Dialogue Synthesis and Summarization Capabilities for Few-Shot Dialogue Summarization</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      In this work, we propose Mutual Reinforcing Data Synthesis (MRDS) within LLMs to improve few-shot dialogue summarization task. Unlike prior methods that require external knowledge, we mutually reinforce the LLM\'s dialogue synthesis and summarization capabilities, allowing them to complement each other during training and enhance overall performances. The dialogue synthesis capability is enhanced by directed preference optimization with preference scoring from summarization capability. The summarization capability is enhanced by the additional high quality dialogue-summary paired data produced by the dialogue synthesis capability. By leveraging the proposed MRDS mechanism, we elicit the internal knowledge of LLM in the format of synthetic data, and use it to augment the few-shot real training dataset. Empirical results demonstrate that our method improves dialogue summarization, achieving a 1.5% increase in ROUGE scores and a 0.3% improvement in BERT scores in few-shot settings. Furthermore, our method attains the highest average scores in human evaluations, surpassing both the pre-trained models and the baselines fine-tuned solely for summarization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17304v1">Child vs. machine language learning: Can the logical structure of human language unleash LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 ISCA/ITG Workshop on Diversity in Large Speech and Language Models
    </div>
    <details class="paper-abstract">
      We argue that human language learning proceeds in a manner that is different in nature from current approaches to training LLMs, predicting a difference in learning biases. We then present evidence from German plural formation by LLMs that confirm our hypothesis that even very powerful implementations produce results that miss aspects of the logic inherent to language that humans have no problem with. We conclude that attention to the different structures of human language and artificial neural networks is likely to be an avenue to improve LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04420v2">KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 36 pages. Code: https://github.com/cmd2001/KVTuner
    </div>
    <details class="paper-abstract">
      KV cache quantization can improve Large Language Models (LLMs) inference throughput and latency in long contexts and large batch-size scenarios while preserving LLMs effectiveness. However, current methods have three unsolved issues: overlooking layer-wise sensitivity to KV cache quantization, high overhead of online fine-grained decision-making, and low flexibility to different LLMs and constraints. Therefore, we thoroughly analyze the inherent correlation of layer-wise transformer attention patterns to KV cache quantization errors and study why key cache is more important than value cache for quantization error reduction. We further propose a simple yet effective framework KVTuner to adaptively search for the optimal hardware-friendly layer-wise KV quantization precision pairs for coarse-grained KV cache with multi-objective optimization and directly utilize the offline searched configurations during online inference. To reduce the computational cost of offline calibration, we utilize the intra-layer KV precision pair pruning and inter-layer clustering to reduce the search space. Experimental results show that we can achieve nearly lossless 3.25-bit mixed precision KV cache quantization for LLMs like Llama-3.1-8B-Instruct and 4.0-bit for sensitive models like Qwen2.5-7B-Instruct on mathematical reasoning tasks. The maximum inference throughput can be improved by 38.3% compared with KV8 quantization over various context lengths. Our code and searched configurations are available at https://github.com/cmd2001/KVTuner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17298v1">Delta Decompression for MoE-based LLMs Compression</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures in large language models (LLMs) achieve exceptional performance, but face prohibitive storage and memory requirements. To address these challenges, we present $D^2$-MoE, a new delta decompression compressor for reducing the parameters of MoE LLMs. Based on observations of expert diversity, we decompose their weights into a shared base weight and unique delta weights. Specifically, our method first merges each expert's weight into the base weight using the Fisher information matrix to capture shared components. Then, we compress delta weights through Singular Value Decomposition (SVD) by exploiting their low-rank properties. Finally, we introduce a semi-dynamical structured pruning strategy for the base weights, combining static and dynamic redundancy analysis to achieve further parameter reduction while maintaining input adaptivity. In this way, our $D^2$-MoE successfully compact MoE LLMs to high compression ratios without additional training. Extensive experiments highlight the superiority of our approach, with over 13% performance gains than other compressors on Mixtral|Phi-3.5|DeepSeek|Qwen2 MoE LLMs at 40$\sim$60% compression rates. Codes are available in https://github.com/lliai/D2MoE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17282v1">Capability Instruction Tuning: A New Paradigm for Dynamic LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 AAAI 2025; Project Page: https://cit-llm-routing.github.io
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated human-like instruction-following abilities, particularly those exceeding 100 billion parameters. The combined capability of some smaller, resource-friendly LLMs can address most of the instructions that larger LLMs excel at. In this work, we explore how to route the best-performing LLM for each instruction to achieve better overall performance. We develop a new paradigm, constructing capability instructions with model capability representation, user instruction, and performance inquiry prompts to assess the performance. To learn from capability instructions, we introduce a new end-to-end framework called Model Selection with Aptitude Test (Model-SAT), which generates positive and negative samples based on what different models perform well or struggle with. Model-SAT uses a model capability encoder that extends its model representation to a lightweight LLM. Our experiments show that Model-SAT understands the performance dimensions of candidate models and provides the probabilities of their capability to handle various instructions. Additionally, during deployment, a new model can quickly infer its aptitude test results across 50 tasks, each with 20 shots. Model-SAT performs state-of-the-art model routing without candidate inference and in real-world new model-released scenarios. The code is available at https://github.com/Now-Join-Us/CIT-LLM-Routing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20296v2">PersonalLLM: Tailoring LLMs to Individual Preferences</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 28 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As LLMs become capable of complex tasks, there is growing potential for personalized interactions tailored to the subtle and idiosyncratic preferences of the user. We present a public benchmark, PersonalLLM, focusing on adapting LLMs to provide maximal benefits for a particular user. Departing from existing alignment benchmarks that implicitly assume uniform preferences, we curate open-ended prompts paired with many high-quality answers over which users would be expected to display heterogeneous latent preferences. Instead of persona-prompting LLMs based on high-level attributes (e.g., user's race or response length), which yields homogeneous preferences relative to humans, we develop a method that can simulate a large user base with diverse preferences from a set of pre-trained reward models. Our dataset and generated personalities offer an innovative testbed for developing personalization algorithms that grapple with continual data sparsity--few relevant feedback from the particular user--by leveraging historical data from other (similar) users. We explore basic in-context learning and meta-learning baselines to illustrate the utility of PersonalLLM and highlight the need for future methodological development. Our dataset is available at https://huggingface.co/datasets/namkoong-lab/PersonalLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17262v1">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 21 pages,6 figures
    </div>
    <details class="paper-abstract">
      The rapid advancements in computing dramatically increase the scale and cost of training Large Language Models (LLMs). Accurately predicting downstream task performance prior to model training is crucial for efficient resource allocation, yet remains challenging due to two primary constraints: (1) the "emergence phenomenon", wherein downstream performance metrics become meaningful only after extensive training, which limits the ability to use smaller models for prediction; (2) Uneven task difficulty distributions and the absence of consistent scaling laws, resulting in substantial metric variability. Existing performance prediction methods suffer from limited accuracy and reliability, thereby impeding the assessment of potential LLM capabilities. To address these challenges, we propose a Clustering-On-Difficulty (COD) downstream performance prediction framework. COD first constructs a predictable support subset by clustering tasks based on difficulty features, strategically excluding non-emergent and non-scalable clusters. The scores on the selected subset serve as effective intermediate predictors of downstream performance on the full evaluation set. With theoretical support, we derive a mapping function that transforms performance metrics from the predictable subset to the full evaluation set, thereby ensuring accurate extrapolation of LLM downstream performance. The proposed method has been applied to predict performance scaling for a 70B LLM, providing actionable insights for training resource allocation and assisting in monitoring the training process. Notably, COD achieves remarkable predictive accuracy on the 70B LLM by leveraging an ensemble of small models, demonstrating an absolute mean deviation of 1.36% across eight important LLM evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17216v1">Making LLMs Reason? The Intermediate Language Problem in Neurosymbolic Approaches</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Logical reasoning tasks manifest themselves as a challenge to Large Language Models (LLMs). Neurosymbolic approaches use LLMs to translate logical reasoning problems formulated in natural language into a formal intermediate language. Subsequently, the usage of symbolic reasoners yields reliable solving thereof. However, LLMs often fail in translation due to poorly chosen intermediate languages. We introduce the intermediate language problem, which is the problem of choosing a suitable formal language representation for neurosymbolic approaches. Theoretically, we argue that its origins lie in the inability of LLMs to distinguish syntax from semantics and the relative independence of the problem from its representation. We showcase its existence experimentally by contrasting two intermediate languages, Answer Set Programming and the Python Knowledge Engine. In addition, we demonstrate the effects of varying degrees of supplementary context information. Our results show a maximum difference in overall-accuracy of 53.20% and 49.26% in execution-accuracy. When using the GPT4o-mini LLM we beat the state-of-the-art in overall-accuracy on the ProntoQA dataset by 21.20% and by 50.50% on the ProofWriter dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17214v1">CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in many tasks but struggle to accurately quantify uncertainty in their generated responses. This limitation makes it challenging to detect misinformation and ensure reliable decision-making. Existing uncertainty quantification (UQ) methods for LLMs are primarily prompt-wise rather than response-wise, often requiring multiple response samples, which incurs high computational costs. Moreover, LLMs have been shown to be overconfident, particularly when using reasoning steps to derive their answers. In this work, we propose CoT-UQ, a response-wise UQ framework that integrates LLMs' inherent reasoning capabilities through Chain-of-Thought (CoT) into the UQ process. CoT-UQ captures critical information during inference by extracting keywords from each reasoning step and assessing their importance to the final answer. This key reasoning information is then aggregated to produce a final uncertainty estimate. We conduct extensive experiments based on LLaMA Family with model sizes varying from 8B to 13B across logical and mathematical reasoning tasks. Experimental results demonstrate that CoT-UQ significantly outperforms existing UQ methods, achieving an average improvement of 5.9% AUROC compared to current UQ methods. The code is available at: https://github.com/ZBox1005/CoT-UQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17187v1">Evaluating Expert Contributions in a MoE LLM for Quiz-Based Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 preprint, short paper
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) with Mixture of Experts (MoE) layers have gained significant attention. Currently, state-of-the-art LLMs utilize this architecture. There is a substantial amount of research on how to train such models and how to select hyperparameters for this architecture. However, there is a lack of studies focusing on post-evaluation analysis of MoE layer properties. In this paper, we take a first step toward closing this gap by evaluating expert contributions on the quiz-based MMLU benchmark. We show that most experts were never activated during inference on this benchmark. Additionally, the output distribution of gating networks is much closer to uniform than sparse. Finally, we demonstrate that the average performance of some experts within the same layer varies significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08090v2">Multilingual LLMs Inherently Reward In-Language Time-Sensitive Semantic Alignment for Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The unwavering disparity in labeled resources between resource-rich languages and those considered low-resource remains a significant impediment for Large Language Models (LLMs). Recent strides in cross-lingual in-context learning (X-ICL), mainly through semantically aligned examples retrieved from multilingual pre-trained transformers, have shown promise in mitigating this issue. However, our investigation reveals that LLMs intrinsically reward in-language semantically aligned cross-lingual instances over direct cross-lingual semantic alignments, with a pronounced disparity in handling time-sensitive queries in the X-ICL setup. Such queries demand sound temporal reasoning ability from LLMs, yet the advancements have predominantly focused on English. This study aims to bridge this gap by improving temporal reasoning capabilities in low-resource languages. To this end, we introduce mTEMPREASON, a temporal reasoning dataset aimed at the varied degrees of low-resource languages and propose Cross-Lingual Time-Sensitive Semantic Alignment (CLiTSSA), a novel method to improve temporal reasoning in these contexts. To facilitate this, we construct an extension of mTEMPREASON comprising pairs of parallel cross-language temporal queries along with their anticipated in-language semantic similarity scores. Our empirical evidence underscores the superior performance of CLiTSSA compared to established baselines across three languages -- Romanian, German, and French, encompassing three temporal tasks and including a diverse set of four contemporaneous LLMs. This marks a significant step forward in addressing resource disparity in the context of temporal reasoning across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18596v2">DeltaLLM: Compress LLMs with Low-Rank Deltas between Shared Weights</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      We introduce DeltaLLM, a new post-training compression technique to reduce the memory footprint of LLMs. We propose an alternative way of structuring LLMs with weight sharing between layers in subsequent Transformer blocks, along with additional low-rank difference matrices between them. For training, we adopt the progressing module replacement method and show that the lightweight training of the low-rank modules with approximately 30M-40M tokens is sufficient to achieve performance on par with LLMs of comparable sizes trained from scratch. We release the resultant models, DeltaLLAMA and DeltaPHI, with a 12% parameter reduction, retaining 90% of the performance of the base Llama and Phi models on common knowledge and reasoning benchmarks. Our method also outperforms compression techniques JointDrop, LaCo, ShortGPT and SliceGPT with the same number of parameters removed. For example, DeltaPhi 2.9B with a 24% reduction achieves similar average zero-shot accuracies as recovery fine-tuned SlicedPhi 3.3B with a 12% reduction, despite being approximately 400M parameters smaller with no fine-tuning applied. This work provides new insights into LLM architecture design and compression methods when storage space is critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15294v2">Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. We propose Round Attention, a novel round-level attention mechanism that only recalls and computes the KV cache of the most relevant rounds. The experiments show that our method saves 55\% memory usage without compromising model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17139v1">CodeSwift: Accelerating LLM Inference for Efficient Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Code generation is a latency-sensitive task that demands high timeliness, but the autoregressive decoding mechanism of Large Language Models (LLMs) leads to poor inference efficiency. Existing LLM inference acceleration methods mainly focus on standalone functions using only built-in components. Moreover, they treat code like natural language sequences, ignoring its unique syntax and semantic characteristics. As a result, the effectiveness of these approaches in code generation tasks remains limited and fails to align with real-world programming scenarios. To alleviate this issue, we propose CodeSwift, a simple yet highly efficient inference acceleration approach specifically designed for code generation, without comprising the quality of the output. CodeSwift constructs a multi-source datastore, providing access to both general and project-specific knowledge, facilitating the retrieval of high-quality draft sequences. Moreover, CodeSwift reduces retrieval cost by controlling retrieval timing, and enhances efficiency through parallel retrieval and a context- and LLM preference-aware cache. Experimental results show that CodeSwift can reach up to 2.53x and 2.54x speedup compared to autoregressive decoding in repository-level and standalone code generation tasks, respectively, outperforming state-of-the-art inference acceleration approaches by up to 88%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17091v1">WildFrame: Comparing Framing in Humans and LLMs on Naturally Occurring Texts</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Humans are influenced by how information is presented, a phenomenon known as the framing effect. Previous work has shown that LLMs may also be susceptible to framing but has done so on synthetic data and did not compare to human behavior. We introduce WildFrame, a dataset for evaluating LLM responses to positive and negative framing, in naturally-occurring sentences, and compare humans on the same data. WildFrame consists of 1,000 texts, first selecting real-world statements with clear sentiment, then reframing them in either positive or negative light, and lastly, collecting human sentiment annotations. By evaluating eight state-of-the-art LLMs on WildFrame, we find that all models exhibit framing effects similar to humans ($r\geq0.57$), with both humans and models being more influenced by positive rather than negative reframing. Our findings benefit model developers, who can either harness framing or mitigate its effects, depending on the downstream application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.05806v3">CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Ongoing work; project website is available at https://zjunlp.github.io/project/CKnowEdit code and dataset are available at https://github.com/zjunlp/EasyEdit
    </div>
    <details class="paper-abstract">
      Chinese, as a linguistic system rich in depth and complexity, is characterized by distinctive elements such as ancient poetry, proverbs, idioms, and other cultural constructs. However, current Large Language Models (LLMs) face limitations in these specialized domains, highlighting the need for the development of comprehensive datasets that can assess, continuously update, and progressively improve these culturally-grounded linguistic competencies through targeted training optimizations. To address this gap, we introduce CKnowEdit, the first-ever Chinese knowledge editing dataset designed to correct linguistic, factual, and logical errors in LLMs. We collect seven types of knowledge from a wide range of sources, including classical texts, idioms, and content from Baidu Tieba Ruozhiba, taking into account the unique polyphony, antithesis, and logical structures inherent in the Chinese language. By analyzing this dataset, we highlight the challenges current LLMs face in mastering Chinese. Furthermore, our evaluation of state-of-the-art knowledge editing techniques reveals opportunities to advance the correction of Chinese knowledge. Code and dataset are available at https://github.com/zjunlp/EasyEdit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v2">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/bytedance-research/BFS-Prover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17026v1">Understanding the Uncertainty of LLM Explanations: A Perspective Based on Reasoning Topology</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding the uncertainty in large language model (LLM) explanations is important for evaluating their faithfulness and reasoning consistency, and thus provides insights into the reliability of LLM's output regarding a question. In this work, we propose a novel framework that quantifies uncertainty in LLM explanations through a reasoning topology perspective. By designing a structural elicitation strategy, we guide the LLMs to frame the explanations of an answer into a graph topology. This process decomposes the explanations into the knowledge related sub-questions and topology-based reasoning structures, which allows us to quantify uncertainty not only at the semantic level but also from the reasoning path. It further brings convenience to assess knowledge redundancy and provide interpretable insights into the reasoning process. Our method offers a systematic way to interpret the LLM reasoning, analyze limitations, and provide guidance for enhancing robustness and faithfulness. This work pioneers the use of graph-structured uncertainty measurement in LLM explanations and demonstrates the potential of topology-based quantification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17011v1">Predicting Liquidity-Aware Bond Yields using Causal GANs and Deep Reinforcement Learning with LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Financial bond yield forecasting is challenging due to data scarcity, nonlinear macroeconomic dependencies, and evolving market conditions. In this paper, we propose a novel framework that leverages Causal Generative Adversarial Networks (CausalGANs) and Soft Actor-Critic (SAC) reinforcement learning (RL) to generate high-fidelity synthetic bond yield data for four major bond categories (AAA, BAA, US10Y, Junk). By incorporating 12 key macroeconomic variables, we ensure statistical fidelity by preserving essential market properties. To transform this market dependent synthetic data into actionable insights, we employ a finetuned Large Language Model (LLM) Qwen2.5-7B that generates trading signals (BUY/HOLD/SELL), risk assessments, and volatility projections. We use automated, human and LLM evaluations, all of which demonstrate that our framework improves forecasting performance over existing methods, with statistical validation via predictive accuracy, MAE evaluation(0.103%), profit/loss evaluation (60% profit rate), LLM evaluation (3.37/5) and expert assessments scoring 4.67 out of 5. The reinforcement learning-enhanced synthetic data generation achieves the least Mean Absolute Error of 0.103, demonstrating its effectiveness in replicating real-world bond market dynamics. We not only enhance data-driven trading strategies but also provides a scalable, high-fidelity synthetic financial data pipeline for risk & volatility management and investment decision-making. This work establishes a bridge between synthetic data generation, LLM driven financial forecasting, and language model evaluation, contributing to AI-driven financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10400v3">Reinforcement Learning Enhanced LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) enhanced large language models (LLMs), particularly exemplified by DeepSeek-R1, have exhibited outstanding performance. Despite the effectiveness in improving LLM capabilities, its implementation remains highly complex, requiring complex algorithms, reward modeling strategies, and optimization techniques. This complexity poses challenges for researchers and practitioners in developing a systematic understanding of RL-enhanced LLMs. Moreover, the absence of a comprehensive survey summarizing existing research on RL-enhanced LLMs has limited progress in this domain, hindering further advancements. In this work, we are going to make a systematic review of the most up-to-date state of knowledge on RL-enhanced LLMs, attempting to consolidate and analyze the rapidly growing research in this field, helping researchers understand the current challenges and advancements. Specifically, we (1) detail the basics of RL; (2) introduce popular RL-enhanced LLMs; (3) review researches on two widely-used reward model-based RL techniques: Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning from AI Feedback (RLAIF); and (4) explore Direct Preference Optimization (DPO), a set of methods that bypass the reward model to directly use human preference data for aligning LLM outputs with human expectations. We will also point out current challenges and deficiencies of existing methods and suggest some avenues for further improvements. Project page of this work can be found at https://github.com/ShuheWang1998/Reinforcement-Learning-Enhanced-LLMs-A-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11022v4">DynamicNER: A Dynamic, Multilingual, and Fine-Grained Dataset for LLM-based Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      With the advancement of Large Language Models (LLMs), more and more researchers apply LLMs for Named Entity Recognition (NER) methods, bringing vitality to this classical Natural Language Processing task. However, existing datasets are designed for traditional machine learning methods, inadequate for LLM-based methods in terms of corpus selection, entity categorization, and design logic. This limitation leads to less effective evaluation and model fine-tuning. To address this issue, we propose DynamicNER, the first NER dataset specifically designed for LLMs and with dynamic categorization, transcending the limitations of fixed categorization in existing datasets. It is also multi-lingual and multi-granular, covering 8 languages and 155 entity types, with corpus spanning multiple specialized domains. Furthermore, in response to the limitations demonstrated by existing LLM-based methods during DynamicNER testing, we develop CascadeNER, a novel NER method based on a two-stage strategy and lightweight LLMs, addressing the problems in current methods. Experiments show that DynamicNER is an effective benchmark for LLM-based NER methods, and CascadeNER outperforms existing methods with fewer computational resources. Our work is opened at https://github.com/CascadeNER/CascadeNER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16963v1">Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 15 pages, 17 figures, accepted by HPCA 2025
    </div>
    <details class="paper-abstract">
      The billion-scale Large Language Models (LLMs) need deployment on expensive server-grade GPUs with large-storage HBMs and abundant computation capability. As LLM-assisted services become popular, achieving cost-effective LLM inference on budget-friendly hardware becomes the trend. Extensive researches relocate LLM parameters from expensive GPUs to host memory. However, the restricted bandwidth between the host and GPU memory limits the inference performance. This work introduces Hermes, a budget-friendly system that leverages the near-data processing (NDP) within commodity DRAM DIMMs to enhance the performance of a single consumer-grade GPU, achieving efficient LLM inference. The inherent activation sparsity in LLMs naturally divides weight parameters into two categories, termed ``hot" and ``cold" neurons, respectively. Hot neurons, which consist of only approximately 20\% of all weight parameters, account for 80\% of the total computational load, while cold neurons make up the other 80\% of parameters but are responsible for just 20\% of the computational load. Therefore, we propose a heterogeneous computing strategy: mapping hot neurons to a single computation-efficient GPU, while offloading cold neurons to NDP-DIMMs, which offer large memory size but limited computation capabilities. Meanwhile, the dynamic nature of activation sparsity needs a real-time partition of hot/cold neurons and adaptive remapping of cold neurons across multiple NDP-DIMM modules. Therefore, we introduce a lightweight predictor optimizing real-time neuron partition and adjustment between GPU and NDP-DIMMs. We also utilize a window-based online scheduling mechanism to maintain load balance among NDP-DIMM modules. Hermes facilitates the deployment of LLaMA2-70B on consumer-grade hardware at 13.75 tokens/s and realizes an average 75.24$\times$ speedup over the state-of-the-art offloading-based inference system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15229v2">User Experience with LLM-powered Conversational Recommendation Systems: A Case of Music Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) now allows users to actively interact with conversational recommendation systems (CRS) and build their own personalized recommendation services tailored to their unique needs and goals. This experience offers users a significantly higher level of controllability compared to traditional RS, enabling an entirely new dimension of recommendation experiences. Building on this context, this study explored the unique experiences that LLM-powered CRS can provide compared to traditional RS. Through a three-week diary study with 12 participants using custom GPTs for music recommendations, we found that LLM-powered CRS can (1) help users clarify implicit needs, (2) support unique exploration, and (3) facilitate a deeper understanding of musical preferences. Based on these findings, we discuss the new design space enabled by LLM-powered CRS and highlight its potential to support more personalized, user-driven recommendation experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21271v3">EoRA: Training-free Compensation for Compressed LLM with Eigenspace Low-Rank Approximation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      In this work, we re-formulate the model compression problem into the customized compensation problem: Given a compressed model, we aim to introduce residual low-rank paths to compensate for compression errors under customized requirements from users (e.g., tasks, compression ratios), resulting in greater flexibility in balancing accuracy and overhead(inference and model size) without being bound to fixed compression formats. However, naively applying SVD to derive residual paths causes suboptimal utilization of the low-rank representation capacity. Instead, we propose Training-free Eigenspace Low-Rank Approximation (EoRA), a method that directly minimizes compression-induced errors without requiring gradient-based training, achieving fast optimization in minutes using a small amount of calibration data. EoRA projects compression errors into the eigenspace of input activations, leveraging eigenvalues to effectively prioritize the reconstruction of high-importance error components. Moreover, EoRA can be seamlessly integrated with fine-tuning and quantization to further improve effectiveness and efficiency. EoRA consistently outperforms previous methods in compensating errors for compressed LLaMA2/3 models on various tasks, such as language generation, commonsense reasoning, and math reasoning tasks (e.g., 31.31%/12.88% and 9.69% improvements on ARC-Easy/ARC-Challenge and MathQA when compensating LLaMA3-8B that is quantized to 4-bit and pruned to 2:4 sparsity). EoRA offers a scalable, training-free solution to compensate for compression errors, making it a powerful tool to deploy LLMs more flexibly. Code is available at https://github.com/NVlabs/EoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16924v1">FilterLLM: Text-To-Distribution LLM for Billion-Scale Cold-Start Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based cold-start recommendation systems continue to face significant computational challenges in billion-scale scenarios, as they follow a "Text-to-Judgment" paradigm. This approach processes user-item content pairs as input and evaluates each pair iteratively. To maintain efficiency, existing methods rely on pre-filtering a small candidate pool of user-item pairs. However, this severely limits the inferential capabilities of LLMs by reducing their scope to only a few hundred pre-filtered candidates. To overcome this limitation, we propose a novel "Text-to-Distribution" paradigm, which predicts an item's interaction probability distribution for the entire user set in a single inference. Specifically, we present FilterLLM, a framework that extends the next-word prediction capabilities of LLMs to billion-scale filtering tasks. FilterLLM first introduces a tailored distribution prediction and cold-start framework. Next, FilterLLM incorporates an efficient user-vocabulary structure to train and store the embeddings of billion-scale users. Finally, we detail the training objectives for both distribution prediction and user-vocabulary construction. The proposed framework has been deployed on the Alibaba platform, where it has been serving cold-start recommendations for two months, processing over one billion cold items. Extensive experiments demonstrate that FilterLLM significantly outperforms state-of-the-art methods in cold-start recommendation tasks, achieving over 30 times higher efficiency. Furthermore, an online A/B test validates its effectiveness in billion-scale recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16895v1">Unlocking Scientific Concepts: How Effective Are LLM-Generated Analogies for Student Understanding and Classroom Practice?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 19 pages, conditionally accepted by CHI 2025
    </div>
    <details class="paper-abstract">
      Teaching scientific concepts is essential but challenging, and analogies help students connect new concepts to familiar ideas. Advancements in large language models (LLMs) enable generating analogies, yet their effectiveness in education remains underexplored. In this paper, we first conducted a two-stage study involving high school students and teachers to assess the effectiveness of LLM-generated analogies in biology and physics through a controlled in-class test and a classroom field study. Test results suggested that LLM-generated analogies could enhance student understanding particularly in biology, but require teachers' guidance to prevent over-reliance and overconfidence. Classroom experiments suggested that teachers could refine LLM-generated analogies to their satisfaction and inspire new analogies from generated ones, encouraged by positive classroom feedback and homework performance boosts. Based on findings, we developed and evaluated a practical system to help teachers generate and refine teaching analogies. We discussed future directions for developing and evaluating LLM-supported teaching and learning by analogy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12913v2">GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to BF16-based fine-tuning while significantly reducing 1.85x memory usage. Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16892v1">Applying LLMs to Active Learning: Towards Cost-Efficient Cross-Task Text Classification without Manually Labeled Data</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Statement in Accordance with IEEE Preprint Policy: This work is intended for submission to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Machine learning-based classifiers have been used for text classification, such as sentiment analysis, news classification, and toxic comment classification. However, supervised machine learning models often require large amounts of labeled data for training, and manual annotation is both labor-intensive and requires domain-specific knowledge, leading to relatively high annotation costs. To address this issue, we propose an approach that integrates large language models (LLMs) into an active learning framework. Our approach combines the Robustly Optimized BERT Pretraining Approach (RoBERTa), Generative Pre-trained Transformer (GPT), and active learning, achieving high cross-task text classification performance without the need for any manually labeled data. Furthermore, compared to directly applying GPT for classification tasks, our approach retains over 93% of its classification performance while requiring only approximately 6% of the computational time and monetary cost, effectively balancing performance and resource efficiency. These findings provide new insights into the efficient utilization of LLMs and active learning algorithms in text classification tasks, paving the way for their broader application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09642v2">Krutrim LLM: Multilingual Foundational Model for over a Billion People</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      India is a diverse society with unique challenges in developing AI systems, including linguistic diversity, oral traditions, data accessibility, and scalability. Existing foundation models are primarily trained on English, limiting their effectiveness for India's population. Indic languages comprise only 1 percent of Common Crawl corpora despite India representing 18 percent of the global population, leading to linguistic biases. Thousands of regional languages, dialects, and code mixing create additional representation challenges due to sparse training data. We introduce Krutrim LLM, a 2 trillion token multilingual model designed for India's linguistic landscape. It incorporates the largest known Indic dataset, mitigating data scarcity and ensuring balanced performance across dialects. Krutrim outperforms or matches state-of-the-art models on Indic benchmarks while maintaining competitive English performance. Despite being significantly smaller in training flops, Krutrim LLM matches or exceeds models like LLAMA-2 on 10 out of 16 tasks, with an average score of 0.57 versus 0.55. This evidences Krutrim's flexible multilingual fluency across diverse linguistic contexts. Krutrim is integrated with real-time search to improve factual accuracy in conversational AI applications. This enhances accessibility for over 1 billion users worldwide. Through intentional design choices addressing data imbalances, Krutrim LLM signifies meaningful progress in building ethical, globally representative AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16879v1">A Multi-LLM-Agent-Based Framework for Economic and Public Policy Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      This paper pioneers a novel approach to economic and public policy analysis by leveraging multiple Large Language Models (LLMs) as heterogeneous artificial economic agents. We first evaluate five LLMs' economic decision-making capabilities in solving two-period consumption allocation problems under two distinct scenarios: with explicit utility functions and based on intuitive reasoning. While previous research has often simulated heterogeneity by solely varying prompts, our approach harnesses the inherent variations in analytical capabilities across different LLMs to model agents with diverse cognitive traits. Building on these findings, we construct a Multi-LLM-Agent-Based (MLAB) framework by mapping these LLMs to specific educational groups and corresponding income brackets. Using interest-income taxation as a case study, we demonstrate how the MLAB framework can simulate policy impacts across heterogeneous agents, offering a promising new direction for economic and public policy analysis by leveraging LLMs' human-like reasoning capabilities and computational power.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10593v3">An Efficient Sign Language Translation Using Spatial Configuration and Motion Dynamics with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Accepted to NAACL 2025 main
    </div>
    <details class="paper-abstract">
      Gloss-free Sign Language Translation (SLT) converts sign videos directly into spoken language sentences without relying on glosses. Recently, Large Language Models (LLMs) have shown remarkable translation performance in gloss-free methods by harnessing their powerful natural language generation capabilities. However, these methods often rely on domain-specific fine-tuning of visual encoders to achieve optimal results. By contrast, this paper emphasizes the importance of capturing the spatial configurations and motion dynamics inherent in sign language. With this in mind, we introduce Spatial and Motion-based Sign Language Translation (SpaMo), a novel LLM-based SLT framework. The core idea of SpaMo is simple yet effective. We first extract spatial and motion features using off-the-shelf visual encoders and then input these features into an LLM with a language prompt. Additionally, we employ a visual-text alignment process as a warm-up before the SLT supervision. Our experiments demonstrate that SpaMo achieves state-of-the-art performance on two popular datasets, PHOENIX14T and How2Sign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13522v2">INDIC QA BENCHMARK: A Multilingual Benchmark to Evaluate Question Answering capability of LLMs for Indic Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) perform well on unseen tasks in English, but their abilities in non English languages are less explored due to limited benchmarks and training data. To bridge this gap, we introduce the Indic QA Benchmark, a large dataset for context grounded question answering in 11 major Indian languages, covering both extractive and abstractive tasks. Evaluations of multilingual LLMs, including instruction finetuned versions, revealed weak performance in low resource languages due to a strong English language bias in their training data. We also investigated the Translate Test paradigm,where inputs are translated to English for processing and the results are translated back into the source language for output. This approach outperformed multilingual LLMs, particularly in low resource settings. By releasing Indic QA, we aim to promote further research into LLMs question answering capabilities in low resource languages. This benchmark offers a critical resource to address existing limitations and foster multilingual understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16852v1">Improving LLM General Preference Alignment via Optimistic Online Mirror Descent</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning from human feedback (RLHF) has demonstrated remarkable effectiveness in aligning large language models (LLMs) with human preferences. Many existing alignment approaches rely on the Bradley-Terry (BT) model assumption, which assumes the existence of a ground-truth reward for each prompt-response pair. However, this assumption can be overly restrictive when modeling complex human preferences. In this paper, we drop the BT model assumption and study LLM alignment under general preferences, formulated as a two-player game. Drawing on theoretical insights from learning in games, we integrate optimistic online mirror descent into our alignment framework to approximate the Nash policy. Theoretically, we demonstrate that our approach achieves an $O(T^{-1})$ bound on the duality gap, improving upon the previous $O(T^{-1/2})$ result. More importantly, we implement our method and show through experiments that it outperforms state-of-the-art RLHF algorithms across multiple representative benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14866v4">PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs.In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16362v2">AgentFL: Scaling LLM-based Fault Localization to Project-Level Context</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 Added a comment to refer to the published version
    </div>
    <details class="paper-abstract">
      Fault Localization (FL) is an essential step during the debugging process. With the strong capabilities of code comprehension, the recent Large Language Models (LLMs) have demonstrated promising performance in diagnosing bugs in the code. Nevertheless, due to LLMs' limited performance in handling long contexts, existing LLM-based fault localization remains on localizing bugs within a small code scope (i.e., a method or a class), which struggles to diagnose bugs for a large code scope (i.e., an entire software system). To address the limitation, this paper presents AgentFL, a multi-agent system based on ChatGPT for automated fault localization. By simulating the behavior of a human developer, AgentFL models the FL task as a three-step process, which involves comprehension, navigation, and confirmation. Within each step, AgentFL hires agents with diversified expertise, each of which utilizes different tools to handle specific tasks. Particularly, we adopt a series of auxiliary strategies such as Test Behavior Tracking, Document-Guided Search, and Multi-Round Dialogue to overcome the challenges in each step. The evaluation on the widely used Defects4J-V1.2.0 benchmark shows that AgentFL can localize 157 out of 395 bugs within Top-1, which outperforms the other LLM-based approaches and exhibits complementarity to the state-of-the-art learning-based techniques. Additionally, we confirm the indispensability of the components in AgentFL with the ablation study and demonstrate the usability of AgentFL through a user study. Finally, the cost analysis shows that AgentFL spends an average of only 0.074 dollars and 97 seconds for a single bug.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16789v1">AlphaAgent: LLM-Driven Alpha Mining with Regularized Exploration to Counteract Alpha Decay</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Alpha mining, a critical component in quantitative investment, focuses on discovering predictive signals for future asset returns in increasingly complex financial markets. However, the pervasive issue of alpha decay, where factors lose their predictive power over time, poses a significant challenge for alpha mining. Traditional methods like genetic programming face rapid alpha decay from overfitting and complexity, while approaches driven by Large Language Models (LLMs), despite their promise, often rely too heavily on existing knowledge, creating homogeneous factors that worsen crowding and accelerate decay. To address this challenge, we propose AlphaAgent, an autonomous framework that effectively integrates LLM agents with ad hoc regularizations for mining decay-resistant alpha factors. AlphaAgent employs three key mechanisms: (i) originality enforcement through a similarity measure based on abstract syntax trees (ASTs) against existing alphas, (ii) hypothesis-factor alignment via LLM-evaluated semantic consistency between market hypotheses and generated factors, and (iii) complexity control via AST-based structural constraints, preventing over-engineered constructions that are prone to overfitting. These mechanisms collectively guide the alpha generation process to balance originality, financial rationale, and adaptability to evolving market conditions, mitigating the risk of alpha decay. Extensive evaluations show that AlphaAgent outperforms traditional and LLM-based methods in mitigating alpha decay across bull and bear markets, consistently delivering significant alpha in Chinese CSI 500 and US S&P 500 markets over the past four years. Notably, AlphaAgent showcases remarkable resistance to alpha decay, elevating the potential for yielding powerful factors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16781v1">MultiOCR-QA: Dataset for Evaluating Robustness of LLMs in Question Answering on Multilingual OCR Texts</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors -- imperfect extraction of the text, including character insertion, deletion and permutation -- can significantly impact downstream tasks like question-answering (QA). In this work, we introduce a multilingual QA dataset MultiOCR-QA, designed to analyze the effects of OCR noise on QA systems' performance. The MultiOCR-QA dataset comprises 60K question-answer pairs covering three languages, English, French, and German. The dataset is curated from OCR-ed old documents, allowing for the evaluation of OCR-induced challenges on question answering. We evaluate MultiOCR-QA on various levels and types of OCR errors to access the robustness of LLMs in handling real-world digitization errors. Our findings show that QA systems are highly prone to OCR induced errors and exhibit performance degradation on noisy OCR text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10757v2">Deep Learning and LLM-based Methods Applied to Stellar Lightcurve Classification</a></div>
    <div class="paper-meta">
      📅 2025-02-24
      | 💬 35 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Light curves serve as a valuable source of information on stellar formation and evolution. With the rapid advancement of machine learning techniques, it can be effectively processed to extract astronomical patterns and information. In this study, we present a comprehensive evaluation of deep-learning and large language model (LLM) based models for the automatic classification of variable star light curves, based on large datasets from the Kepler and K2 missions. Special emphasis is placed on Cepheids, RR Lyrae, and eclipsing binaries, examining the influence of observational cadence and phase distribution on classification precision. Employing AutoDL optimization, we achieve striking performance with the 1D-Convolution+BiLSTM architecture and the Swin Transformer, hitting accuracies of 94\% and 99\% correspondingly, with the latter demonstrating a notable 83\% accuracy in discerning the elusive Type II Cepheids-comprising merely 0.02\% of the total dataset.We unveil StarWhisper LightCurve (LC), an innovative Series comprising three LLM-based models: LLM, multimodal large language model (MLLM), and Large Audio Language Model (LALM). Each model is fine-tuned with strategic prompt engineering and customized training methods to explore the emergent abilities of these models for astronomical data. Remarkably, StarWhisper LC Series exhibit high accuracies around 90\%, significantly reducing the need for explicit feature engineering, thereby paving the way for streamlined parallel data processing and the progression of multifaceted multimodal models in astronomical applications. The study furnishes two detailed catalogs illustrating the impacts of phase and sampling intervals on deep learning classification accuracy, showing that a substantial decrease of up to 14\% in observation duration and 21\% in sampling points can be realized without compromising accuracy by more than 10\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17535v1">The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve?</a></div>
    <div class="paper-meta">
      📅 2025-02-24
    </div>
    <details class="paper-abstract">
      Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16985v2">Unveiling LLM Mechanisms Through Neural ODEs and Control Theory</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      This paper proposes a framework combining Neural Ordinary Differential Equations (Neural ODEs) and robust control theory to enhance the interpretability and control of large language models (LLMs). By utilizing Neural ODEs to model the dynamic evolution of input-output relationships and introducing control mechanisms to optimize output quality, we demonstrate the effectiveness of this approach across multiple question-answer datasets. Experimental results show that the integration of Neural ODEs and control theory significantly improves output consistency and model interpretability, advancing the development of explainable AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08656v2">LLMs Are Biased Towards Output Formats! Systematically Evaluating and Mitigating Output Format Bias of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 NAACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We present the first systematic evaluation examining format bias in performance of large language models (LLMs). Our approach distinguishes between two categories of an evaluation metric under format constraints to reliably and accurately assess performance: one measures performance when format constraints are adhered to, while the other evaluates performance regardless of constraint adherence. We then define a metric for measuring the format bias of LLMs and establish effective strategies to reduce it. Subsequently, we present our empirical format bias evaluation spanning four commonly used categories -- multiple-choice question-answer, wrapping, list, and mapping -- covering 15 widely-used formats. Our evaluation on eight generation tasks uncovers significant format bias across state-of-the-art LLMs. We further discover that improving the format-instruction following capabilities of LLMs across formats potentially reduces format bias. Based on our evaluation findings, we study prompting and fine-tuning with synthesized format data techniques to mitigate format bias. Our methods successfully reduce the variance in ChatGPT's performance among wrapping formats from 235.33 to 0.71 (%$^2$).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12103v3">Does Unlearning Truly Unlearn? A Black Box Evaluation of LLM Unlearning Methods</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language model unlearning aims to remove harmful information that LLMs have learnt to prevent their use for malicious purposes. LLMU and RMU have been proposed as two methods for LLM unlearning, achieving impressive results on unlearning benchmarks. We study in detail the impact of unlearning on LLM performance metrics using the WMDP dataset as well as a new biology dataset we create. We show that unlearning has a notable impact on general model capabilities, with the performance degradation being more significant in general for LLMU. We further test the robustness of the two methods and find that doing 5-shot prompting or rephrasing the question in simple ways can lead to an over ten-fold increase in accuracy on unlearning benchmarks. Finally, we show that training on unrelated data can almost completely recover pre-unlearning performance, demonstrating that these methods fail at truly unlearning. Our methodology serves as an evaluation framework for LLM unlearning methods. The code is available at: https://github.com/JaiDoshi/Knowledge-Erasure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06205v2">Leveraging Edge Intelligence and LLMs to Advance 6G-Enabled Internet of Automated Defense Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 8 pages, 5 figures, accepted to IEEE Internet of Things Magazine
    </div>
    <details class="paper-abstract">
      The evolution of Artificial Intelligence (AI) and its subset Deep Learning (DL), has profoundly impacted numerous domains, including autonomous driving. The integration of autonomous driving in military settings reduces human casualties and enables precise and safe execution of missions in hazardous environments while allowing for reliable logistics support without the risks associated with fatigue-related errors. However, relying on autonomous driving solely requires an advanced decision-making model that is adaptable and optimum in any situation. Considering the presence of numerous interconnected autonomous vehicles in mission-critical scenarios, Ultra-Reliable Low Latency Communication (URLLC) is vital for ensuring seamless coordination, real-time data exchange, and instantaneous response to dynamic driving environments. The advent of 6G strengthens the Internet of Automated Defense Vehicles (IoADV) concept within the realm of Internet of Military Defense Things (IoMDT) by enabling robust connectivity, crucial for real-time data exchange, advanced navigation, and enhanced safety features through IoADV interactions. On the other hand, a critical advancement in this space is using pre-trained Generative Large Language Models (LLMs) for decision-making and communication optimization for autonomous driving. Hence, this work presents opportunities and challenges with a vision of realizing the full potential of these technologies in critical defense applications, especially through the advancement of IoADV and its role in enhancing autonomous military operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16399v1">Ensemble ToT of LLMs and Its Application to Automatic Grading System for Supporting Self-Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 33 pages, 25 figures
    </div>
    <details class="paper-abstract">
      Providing students with detailed and timely grading feedback is essential for self-learning. While existing LLM-based grading systems are promising, most of them rely on one single model, which limits their performance. To address this, we propose Ensemble Tree-of-Thought (ToT), a framework that enhances LLM outputs by integrating multiple models. Using this framework, we develop a grading system. Ensemble ToT follows three steps: (1) analyzing LLM performance, (2) generating candidate answers, and (3) refining them into a final result. Based on this, our grading system first evaluates the grading tendencies of LLMs, then generates multiple results, and finally integrates them via a simulated debate. Experimental results demonstrate our approach's ability to provide accurate and explainable grading by effectively coordinating multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16395v1">An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated potential for data science tasks via code generation. However, the exploratory nature of data science, alongside the stochastic and opaque outputs of LLMs, raise concerns about their reliability. While prior work focuses on benchmarking LLM accuracy, reproducibility remains underexplored, despite being critical to establishing trust in LLM-driven analysis. We propose a novel analyst-inspector framework to automatically evaluate and enforce the reproducibility of LLM-generated data science workflows - the first rigorous approach to the best of our knowledge. Defining reproducibility as the sufficiency and completeness of workflows for reproducing functionally equivalent code, this framework enforces computational reproducibility principles, ensuring transparent, well-documented LLM workflows while minimizing reliance on implicit model assumptions. Using this framework, we systematically evaluate five state-of-the-art LLMs on 1,032 data analysis tasks across three diverse benchmark datasets. We also introduce two novel reproducibility-enhancing prompting strategies. Our results show that higher reproducibility strongly correlates with improved accuracy and reproducibility-enhancing prompts are effective, demonstrating structured prompting's potential to enhance automated data science workflows and enable transparent, robust AI-driven analysis. Our code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18518v1">Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16747v1">SQLong: Enhanced NL2SQL for Longer Contexts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Open-weight large language models (LLMs) have significantly advanced performance in the Natural Language to SQL (NL2SQL) task. However, their effectiveness diminishes when dealing with large database schemas, as the context length increases. To address this limitation, we present SQLong, a novel and efficient data augmentation framework designed to enhance LLM performance in long-context scenarios for the NL2SQL task. SQLong generates augmented datasets by extending existing database schemas with additional synthetic CREATE TABLE commands and corresponding data rows, sampled from diverse schemas in the training data. This approach effectively simulates long-context scenarios during finetuning and evaluation. Through experiments on the Spider and BIRD datasets, we demonstrate that LLMs finetuned with SQLong-augmented data significantly outperform those trained on standard datasets. These imply SQLong's practical implementation and its impact on improving NL2SQL capabilities in real-world settings with complex database schemas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08426v4">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from users' natural language questions (text-to-SQL) remains a long-standing challenge due to the complexities involved in user question understanding, database schema comprehension, and SQL generation. Traditional text-to-SQL systems, which combine human engineering and deep neural networks, have made significant progress. Subsequently, pre-trained language models (PLMs) have been developed for text-to-SQL tasks, achieving promising results. However, as modern databases and user questions grow more complex, PLMs with a limited parameter size often produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which restricts the application of PLM-based systems. Recently, large language models (LLMs) have shown significant capabilities in natural language understanding as model scale increases. Thus, integrating LLM-based solutions can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we provide a comprehensive review of existing LLM-based text-to-SQL studies. Specifically, we offer a brief overview of the technical challenges and evolutionary process of text-to-SQL. Next, we introduce the datasets and metrics designed to evaluate text-to-SQL systems. Subsequently, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we make a summarization and discuss the remaining challenges in this field and suggest expectations for future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16730v1">RapidPen: Fully Automated IP-to-Shell Penetration Testing with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      We present RapidPen, a fully automated penetration testing (pentesting) framework that addresses the challenge of achieving an initial foothold (IP-to-Shell) without human intervention. Unlike prior approaches that focus primarily on post-exploitation or require a human-in-the-loop, RapidPen leverages large language models (LLMs) to autonomously discover and exploit vulnerabilities, starting from a single IP address. By integrating advanced ReAct-style task planning (Re) with retrieval-augmented knowledge bases of successful exploits, along with a command-generation and direct execution feedback loop (Act), RapidPen systematically scans services, identifies viable attack vectors, and executes targeted exploits in a fully automated manner. In our evaluation against a vulnerable target from the Hack The Box platform, RapidPen achieved shell access within 200-400 seconds at a per-run cost of approximately \$0.3-\$0.6, demonstrating a 60\% success rate when reusing prior "success-case" data. These results underscore the potential of truly autonomous pentesting for both security novices and seasoned professionals. Organizations without dedicated security teams can leverage RapidPen to quickly identify critical vulnerabilities, while expert pentesters can offload repetitive tasks and focus on complex challenges. Ultimately, our work aims to make penetration testing more accessible and cost-efficient, thereby enhancing the overall security posture of modern software ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05374v2">Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16696v1">Dynamic LLM Routing and Selection based on User Preferences: Balancing Performance, Cost, and Ethics</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      With the widespread deployment of large language models (LLMs) such as GPT4, BART, and LLaMA, the need for a system that can intelligently select the most suitable model for specific tasks while balancing cost, latency, accuracy, and ethical considerations has become increasingly important. Recognizing that not all tasks necessitate models with over 100 billion parameters, we introduce OptiRoute, an advanced model routing engine designed to dynamically select and route tasks to the optimal LLM based on detailed user-defined requirements. OptiRoute captures both functional (e.g., accuracy, speed, cost) and non-functional (e.g., helpfulness, harmlessness, honesty) criteria, leveraging lightweight task analysis and complexity estimation to efficiently match tasks with the best-fit models from a diverse array of LLMs. By employing a hybrid approach combining k-nearest neighbors (kNN) search and hierarchical filtering, OptiRoute optimizes for user priorities while minimizing computational overhead. This makes it ideal for real-time applications in cloud-based ML platforms, personalized AI services, and regulated industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16690v1">From Text to Space: Mapping Abstract Spatial Models in LLMs during a Grid-World Navigation Task</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      Understanding how large language models (LLMs) represent and reason about spatial information is crucial for building robust agentic systems that can navigate real and simulated environments. In this work, we investigate the influence of different text-based spatial representations on LLM performance and internal activations in a grid-world navigation task. By evaluating models of various sizes on a task that requires navigating toward a goal, we examine how the format used to encode spatial information impacts decision-making. Our experiments reveal that cartesian representations of space consistently yield higher success rates and path efficiency, with performance scaling markedly with model size. Moreover, probing LLaMA-3.1-8B revealed subsets of internal units, primarily located in intermediate layers, that robustly correlate with spatial features, such as the position of the agent in the grid or action correctness, regardless of how that information is represented, and are also activated by unrelated spatial reasoning tasks. This work advances our understanding of how LLMs process spatial information and provides valuable insights for developing more interpretable and robust agentic AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v4">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 Multi-Agent AI in the Real World @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/TradingAgents-AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00581v2">Are the Values of LLMs Structurally Aligned with Humans? A Causal Perspective</a></div>
    <div class="paper-meta">
      📅 2025-02-23
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into critical applications, aligning their behavior with human values presents significant challenges. Current methods, such as Reinforcement Learning from Human Feedback (RLHF), typically focus on a limited set of coarse-grained values and are resource-intensive. Moreover, the correlations between these values remain implicit, leading to unclear explanations for value-steering outcomes. Our work argues that a latent causal value graph underlies the value dimensions of LLMs and that, despite alignment training, this structure remains significantly different from human value systems. We leverage these causal value graphs to guide two lightweight value-steering methods: role-based prompting and sparse autoencoder (SAE) steering, effectively mitigating unexpected side effects. Furthermore, SAE provides a more fine-grained approach to value steering. Experiments on Gemma-2B-IT and Llama3-8B-IT demonstrate the effectiveness and controllability of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16606v1">Reasoning about Affordances: Causal and Compositional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 21 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      With the rapid progress of Large Language Models (LLMs), it becomes increasingly important to understand their abilities and limitations. In two experiments, we investigate the causal and compositional reasoning abilities of LLMs and humans in the domain of object affordances, an area traditionally linked to embodied cognition. The tasks, designed from scratch to avoid data contamination, require decision-makers to select unconventional objects to replace a typical tool for a particular purpose, such as using a table tennis racket to dig a hole. In Experiment 1, we evaluated GPT-3.5 and GPT-4o, finding that GPT-4o, when given chain-of-thought prompting, performed on par with human participants, while GPT-3.5 lagged significantly. In Experiment 2, we introduced two new conditions, Distractor (more object choices, increasing difficulty) and Image (object options presented visually), and evaluated Claude 3 Sonnet and Claude 3.5 Sonnet in addition to the GPT models. The Distractor condition significantly impaired performance across humans and models, although GPT-4o and Claude 3.5 still performed well above chance. Surprisingly, the Image condition had little impact on humans or GPT-4o, but significantly lowered Claude 3.5's accuracy. Qualitative analysis showed that GPT-4o and Claude 3.5 have a stronger ability than their predecessors to identify and flexibly apply causally relevant object properties. The improvement from GPT-3.5 and Claude 3 to GPT-4o and Claude 3.5 suggests that models are increasingly capable of causal and compositional reasoning in some domains, although further mechanistic research is necessary to understand how LLMs reason.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02233v4">Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-23
      | 💬 This version has been accepted and published at EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) like ChatGPT have demonstrated impressive abilities in generating responses based on human instructions. However, their use in the medical field can be challenging due to their lack of specific, in-depth knowledge. In this study, we present a system called LLMs Augmented with Medical Textbooks (LLM-AMT) designed to enhance the proficiency of LLMs in specialized domains. LLM-AMT integrates authoritative medical textbooks into the LLMs' framework using plug-and-play modules. These modules include a Query Augmenter, a Hybrid Textbook Retriever, and a Knowledge Self-Refiner. Together, they incorporate authoritative medical knowledge. Additionally, an LLM Reader aids in contextual understanding. Our experimental results on three medical QA tasks demonstrate that LLMAMT significantly improves response quality, with accuracy gains ranging from 11.6% to 16.6%. Notably, with GPT-4-Turbo as the base model, LLM-AMT outperforms the specialized Med-PaLM 2 model pre-trained on a massive amount of medical corpus by 2-3%. We found that despite being 100x smaller in size, medical textbooks as a retrieval corpus is proven to be a more effective knowledge database than Wikipedia in the medical domain, boosting performance by 7.8%-13.7%.
    </details>
</div>
