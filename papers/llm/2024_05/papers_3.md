# llm - 2024_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05466v2">Poser: Unmasking Alignment Faking LLMs by Manipulating Their Internals</a></div>
    <div class="paper-meta">
      📅 2024-05-11
    </div>
    <details class="paper-abstract">
      Like a criminal under investigation, Large Language Models (LLMs) might pretend to be aligned while evaluated and misbehave when they have a good opportunity. Can current interpretability methods catch these 'alignment fakers?' To answer this question, we introduce a benchmark that consists of 324 pairs of LLMs fine-tuned to select actions in role-play scenarios. One model in each pair is consistently benign (aligned). The other model misbehaves in scenarios where it is unlikely to be caught (alignment faking). The task is to identify the alignment faking model using only inputs where the two models behave identically. We test five detection strategies, one of which identifies 98% of alignment-fakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00971v2">Exploring and Evaluating Hallucinations in LLM-Powered Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-11
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has significantly advanced many applications on software engineering tasks, particularly in code generation. Despite the promising performance, LLMs are prone to generate hallucinations, which means LLMs might produce outputs that deviate from users' intent, exhibit internal inconsistencies, or misalign with the factual knowledge, making the deployment of LLMs potentially risky in a wide range of applications. Existing work mainly focuses on investing the hallucination in the domain of natural language generation (NLG), leaving a gap in understanding the types and extent of hallucinations in the context of code generation. To bridge the gap, we conducted a thematic analysis of the LLM-generated code to summarize and categorize the hallucinations present in it. Our study established a comprehensive taxonomy of hallucinations in LLM-generated code, encompassing 5 primary categories of hallucinations depending on the conflicting objectives and varying degrees of deviation observed in code generation. Furthermore, we systematically analyzed the distribution of hallucinations, exploring variations among different LLMs and their correlation with code correctness. Based on the results, we proposed HalluCode, a benchmark for evaluating the performance of code LLMs in recognizing hallucinations. Hallucination recognition and mitigation experiments with HalluCode and HumanEval show existing LLMs face great challenges in recognizing hallucinations, particularly in identifying their types, and are hardly able to mitigate hallucinations. We believe our findings will shed light on future research about hallucination evaluation, detection, and mitigation, ultimately paving the way for building more effective and reliable code LLMs in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06856v1">Aladdin: Joint Placement and Scaling for SLO-Aware LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-05-11
    </div>
    <details class="paper-abstract">
      The demand for large language model (LLM) inference is gradually dominating the artificial intelligence workloads. Therefore, there is an urgent need for cost-efficient inference serving. Existing work focuses on single-worker optimization and lacks consideration of cluster-level management for both inference queries and computing resources. However, placing requests and managing resources without considering the query features easily causes SLO violations or resource underutilization. Providers are forced to allocate extra computing resources to guarantee user experience, leading to additional serving costs. In this paper we introduce Aladdin, a scheduler that co-adaptively places queries and scales computing resources with SLO awareness. For a stream of inference queries, Aladdin first predicts minimal computing resources and the corresponding serving workers' configuration required to fulfill the SLOs for all queries. Then, it places the queries to each serving worker according to the prefill and decode latency models of batched LLM inference to maximize each worker's utilization. Results show that Aladdin reduces the serving cost of a single model by up to 71% for the same SLO level compared with the baselines, which can be millions of dollars per year.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06840v1">MEIC: Re-thinking RTL Debug Automation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-10
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) for code debugging (e.g., C and Python) is widespread, benefiting from their ability to understand and interpret intricate concepts. However, in the semiconductor industry, utilising LLMs to debug Register Transfer Level (RTL) code is still insufficient, largely due to the underrepresentation of RTL-specific data in training sets. This work introduces a novel framework, Make Each Iteration Count (MEIC), which contrasts with traditional one-shot LLM-based debugging methods that heavily rely on prompt engineering, model tuning, and model training. MEIC utilises LLMs in an iterative process to overcome the limitation of LLMs in RTL code debugging, which is suitable for identifying and correcting both syntax and function errors, while effectively managing the uncertainties inherent in LLM operations. To evaluate our framework, we provide an open-source dataset comprising 178 common RTL programming errors. The experimental results demonstrate that the proposed debugging framework achieves fix rate of 93% for syntax errors and 78% for function errors, with up to 48x speedup in debugging processes when compared with experienced engineers. The Repo. of dataset and code: https://anonymous.4open.science/r/Verilog-Auto-Debug-6E7F/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06835v1">Automating Code Adaptation for MLOps -- A Benchmarking Study on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 The work was completed during 2Q, 3Q of Year 2023, when WizardCoder was the top performing Open source LLM for coding. Newer and better models have emerged since then. The processes and methodologies utilized for this benchmarking can still be utilized for evaluating the current SoTA models
    </div>
    <details class="paper-abstract">
      This paper explores the possibilities of the current generation of Large Language Models for incorporating Machine Learning Operations (MLOps) functionalities into ML training code bases. We evaluate the performance of OpenAI (gpt-3.5-turbo) and WizardCoder (open-source, 15B parameters) models on the automated accomplishment of various MLOps functionalities in different settings. We perform a benchmarking study that assesses the ability of these models to: (1) adapt existing code samples (Inlining) with component-specific MLOps functionality such as MLflow and Weights & Biases for experiment tracking, Optuna for hyperparameter optimization etc., and (2) perform the task of Translation from one component of an MLOps functionality to another, e.g., translating existing GitPython library based version control code to Data Version Control library based. We also propose three different approaches that involve teaching LLMs to comprehend the API documentation of the components as a reference while accomplishing the Translation tasks. In our evaluations, the gpt-3.5-turbo model significantly outperforms WizardCoder by achieving impressive Pass@3 accuracy in model optimization (55% compared to 0% by WizardCoder), experiment tracking (100%, compared to 62.5% by WizardCoder), model registration (92% compared to 42% by WizardCoder) and hyperparameter optimization (83% compared to 58% by WizardCoder) on average, in their best possible settings, showcasing its superior code adaptability performance in complex MLOps tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03998v2">Sketch Then Generate: Providing Incremental User Feedback and Guiding LLM Code Generation through Language-Oriented Code Sketches</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Crafting effective prompts for code generation or editing with Large Language Models (LLMs) is not an easy task. Particularly, the absence of immediate, stable feedback during prompt crafting hinders effective interaction, as users are left to mentally imagine possible outcomes until the code is generated. In response, we introduce Language-Oriented Code Sketching, an interactive approach that provides instant, incremental feedback in the form of code sketches (i.e., incomplete code outlines) during prompt crafting. This approach converts a prompt into a code sketch by leveraging the inherent linguistic structures within the prompt and applying classic natural language processing techniques. The sketch then serves as an intermediate placeholder that not only previews the intended code structure but also guides the LLM towards the desired code, thereby enhancing human-LLM interaction. We conclude by discussing the approach's applicability and future plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06772v1">CANAL -- Cyber Activity News Alerting Language Model: Empirical Approach vs. Expensive LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 Published in 2024 IEEE 3rd International Conference on AI in Cybersecurity (ICAIC), Conference Date: 07-09 February 2024
    </div>
    <details class="paper-abstract">
      In today's digital landscape, where cyber attacks have become the norm, the detection of cyber attacks and threats is critically imperative across diverse domains. Our research presents a new empirical framework for cyber threat modeling, adept at parsing and categorizing cyber-related information from news articles, enhancing real-time vigilance for market stakeholders. At the core of this framework is a fine-tuned BERT model, which we call CANAL - Cyber Activity News Alerting Language Model, tailored for cyber categorization using a novel silver labeling approach powered by Random Forest. We benchmark CANAL against larger, costlier LLMs, including GPT-4, LLaMA, and Zephyr, highlighting their zero to few-shot learning in cyber news classification. CANAL demonstrates superior performance by outperforming all other LLM counterparts in both accuracy and cost-effectiveness. Furthermore, we introduce the Cyber Signal Discovery module, a strategic component designed to efficiently detect emerging cyber signals from news articles. Collectively, CANAL and Cyber Signal Discovery module equip our framework to provide a robust and cost-effective solution for businesses that require agile responses to cyber intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09127v3">Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 Accepted at ICLR 2024 Workshop on Reliable and Responsible Foundation Models
    </div>
    <details class="paper-abstract">
      Uncertainty estimation is a significant issue for current large language models (LLMs) that are generally poorly calibrated and over-confident, especially with reinforcement learning from human feedback (RLHF). Unlike humans, whose decisions and confidences not only stem from intrinsic beliefs but can also be adjusted through daily observations, existing calibration methods for LLMs focus on estimating or eliciting individual confidence without taking full advantage of the "Collective Wisdom": the interaction among multiple LLMs that can collectively improve both accuracy and calibration. In this work, we propose Collaborative Calibration, a post-hoc training-free calibration strategy that leverages the collaborative and expressive capabilities of multiple tool-augmented LLM agents in a simulated group deliberation process. We demonstrate the effectiveness of Collaborative Calibration on generative QA tasks across various domains, showing its potential in harnessing the rationalization of collectively calibrated confidence assessments and improving the reliability of model predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04532v2">QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 The first three authors contribute equally to this project and are listed in the alphabetical order. Yujun Lin leads the quantization algorithm, Haotian Tang and Shang Yang lead the GPU kernels and the serving system. Code is available at https://github.com/mit-han-lab/qserve
    </div>
    <details class="paper-abstract">
      Quantization can accelerate large language model (LLM) inference. Going beyond INT8 quantization, the research community is actively exploring even lower precision, such as INT4. Nonetheless, state-of-the-art INT4 quantization techniques only accelerate low-batch, edge LLM inference, failing to deliver performance gains in large-batch, cloud-based LLM serving. We uncover a critical issue: existing INT4 quantization methods suffer from significant runtime overhead (20-90%) when dequantizing either weights or partial sums on GPUs. To address this challenge, we introduce QoQ, a W4A8KV4 quantization algorithm with 4-bit weight, 8-bit activation, and 4-bit KV cache. QoQ stands for quattuor-octo-quattuor, which represents 4-8-4 in Latin. QoQ is implemented by the QServe inference library that achieves measured speedup. The key insight driving QServe is that the efficiency of LLM serving on GPUs is critically influenced by operations on low-throughput CUDA cores. Building upon this insight, in QoQ algorithm, we introduce progressive quantization that can allow low dequantization overhead in W4A8 GEMM. Additionally, we develop SmoothAttention to effectively mitigate the accuracy degradation incurred by 4-bit KV quantization. In the QServe system, we perform compute-aware weight reordering and take advantage of register-level parallelism to reduce dequantization latency. We also make fused attention memory-bound, harnessing the performance gain brought by KV4 quantization. As a result, QServe improves the maximum achievable serving throughput of Llama-3-8B by 1.2x on A100, 1.4x on L40S; and Qwen1.5-72B by 2.4x on A100, 3.5x on L40S, compared to TensorRT-LLM. Remarkably, QServe on L40S GPU can achieve even higher throughput than TensorRT-LLM on A100. Thus, QServe effectively reduces the dollar cost of LLM serving by 3x. Code is available at https://github.com/mit-han-lab/qserve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06410v1">Potential and Limitations of LLMs in Capturing Structured Semantics: A Case Study on SRL</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 Accepted by ICIC 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) play a crucial role in capturing structured semantics to enhance language understanding, improve interpretability, and reduce bias. Nevertheless, an ongoing controversy exists over the extent to which LLMs can grasp structured semantics. To assess this, we propose using Semantic Role Labeling (SRL) as a fundamental task to explore LLMs' ability to extract structured semantics. In our assessment, we employ the prompting approach, which leads to the creation of our few-shot SRL parser, called PromptSRL. PromptSRL enables LLMs to map natural languages to explicit semantic structures, which provides an interpretable window into the properties of LLMs. We find interesting potential: LLMs can indeed capture semantic structures, and scaling-up doesn't always mirror potential. Additionally, limitations of LLMs are observed in C-arguments, etc. Lastly, we are surprised to discover that significant overlap in the errors is made by both LLMs and untrained humans, accounting for almost 30% of all errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06275v1">Pruning as a Domain-specific LLM Extractor</a></div>
    <div class="paper-meta">
      📅 2024-05-10
      | 💬 NAACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable proficiency across a wide array of NLP tasks. However, the escalation in model size also engenders substantial deployment costs. While few efforts have explored model pruning techniques to reduce the size of LLMs, they mainly center on general or task-specific weights. This leads to suboptimal performance due to lacking specificity on the target domain or generality on different tasks when applied to domain-specific challenges. This work introduces an innovative unstructured dual-pruning methodology, D-Pruner, for domain-specific compression on LLM. It extracts a compressed, domain-specific, and task-agnostic LLM by identifying LLM weights that are pivotal for general capabilities, like linguistic capability and multi-task solving, and domain-specific knowledge. More specifically, we first assess general weight importance by quantifying the error incurred upon their removal with the help of an open-domain calibration dataset. Then, we utilize this general weight importance to refine the training loss, so that it preserves generality when fitting into a specific domain. Moreover, by efficiently approximating weight importance with the refined training loss on a domain-specific calibration dataset, we obtain a pruned model emphasizing generality and specificity. Our comprehensive experiments across various tasks in healthcare and legal domains show the effectiveness of D-Pruner in domain-specific compression. Our code is available at https://github.com/psunlpgroup/D-Pruner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06064v1">LLMs for XAI: Future Directions for Explaining Explanations</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      In response to the demand for Explainable Artificial Intelligence (XAI), we investigate the use of Large Language Models (LLMs) to transform ML explanations into natural, human-readable narratives. Rather than directly explaining ML models using LLMs, we focus on refining explanations computed using existing XAI algorithms. We outline several research directions, including defining evaluation metrics, prompt design, comparing LLM models, exploring further training methods, and integrating external data. Initial experiments and user study suggest that LLMs offer a promising way to enhance the interpretability and usability of XAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06061v1">Supporting Physical Activity Behavior Change with LLM-Based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Physical activity has significant benefits to health, yet large portions of the population remain physically inactive. Mobile health applications show promising potential for low-cost, scalable physical activity promotion, but existing approaches are often insufficiently personalized to a user's context and life circumstances. In this work, we explore the potential for large language model (LLM) based conversational agents to motivate physical activity behavior change. Through formative interviews with 12 health professionals and 10 non-experts, we identify design considerations and opportunities for LLM health coaching. We present GPTCoach, a chatbot that implements an evidence-based health coaching program, uses counseling strategies from motivational interviewing, and can query and visualize health data from a wearable through tool use. We evaluate GPTCoach as a technology probe in a user study with 16 participants. Through quantitive and qualitative analyses, we find promising evidence that GPTCoach can adhere to a health coaching program while adopting a facilitative, supportive, and non-judgmental tone. We find more variable support for GPTCoach's ability to proactively make use of data in ways that foster motivation and empowerment. We conclude with a discussion of our findings, implications for future research, as well as risks and limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05949v1">CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Multimodal Large Language Models (LLMs) have focused primarily on scaling by increasing text-image pair data and enhancing LLMs to improve performance on multimodal tasks. However, these scaling approaches are computationally expensive and overlook the significance of improving model capabilities from the vision side. Inspired by the successful applications of Mixture-of-Experts (MoE) in LLMs, which improves model scalability during training while keeping inference costs similar to those of smaller models, we propose CuMo. CuMo incorporates Co-upcycled Top-K sparsely-gated Mixture-of-experts blocks into both the vision encoder and the MLP connector, thereby enhancing the multimodal LLMs with minimal additional activated parameters during inference. CuMo first pre-trains the MLP blocks and then initializes each expert in the MoE block from the pre-trained MLP block during the visual instruction tuning stage. Auxiliary losses are used to ensure a balanced loading of experts. CuMo outperforms state-of-the-art multimodal LLMs across various VQA and visual-instruction-following benchmarks using models within each model size group, all while training exclusively on open-sourced datasets. The code and model weights for CuMo are open-sourced at https://github.com/SHI-Labs/CuMo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05824v1">Robots Can Feel: LLM-based Framework for Robot Ethical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 The paper is submitted to the IEEE conference
    </div>
    <details class="paper-abstract">
      This paper presents the development of a novel ethical reasoning framework for robots. "Robots Can Feel" is the first system for robots that utilizes a combination of logic and human-like emotion simulation to make decisions in morally complex situations akin to humans. The key feature of the approach is the management of the Emotion Weight Coefficient - a customizable parameter to assign the role of emotions in robot decision-making. The system aims to serve as a tool that can equip robots of any form and purpose with ethical behavior close to human standards. Besides the platform, the system is independent of the choice of the base model. During the evaluation, the system was tested on 8 top up-to-date LLMs (Large Language Models). This list included both commercial and open-source models developed by various companies and countries. The research demonstrated that regardless of the model choice, the Emotions Weight Coefficient influences the robot's decision similarly. According to ANOVA analysis, the use of different Emotion Weight Coefficients influenced the final decision in a range of situations, such as in a request for a dietary violation F(4, 35) = 11.2, p = 0.0001 and in an animal compassion situation F(4, 35) = 8.5441, p = 0.0001. A demonstration code repository is provided at: https://github.com/TemaLykov/robots_can_feel
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10818v3">SlimPajama-DC: Understanding Data Combinations for LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 Technical report. Models at: https://huggingface.co/MBZUAI-LLM/SlimPajama-DC and dataset at: https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC
    </div>
    <details class="paper-abstract">
      This paper aims to understand the impacts of various data combinations (e.g., web text, Wikipedia, GitHub, books) on the pretraining of large language models using SlimPajama. SlimPajama is a rigorously deduplicated, multi-source dataset, which has been refined and further deduplicated to 627B tokens from the extensive 1.2T token RedPajama dataset contributed by Together. We have termed our research as SlimPajama-DC, an empirical analysis designed to uncover fundamental characteristics and best practices associated with employing SlimPajama in the training of large language models. During our research with SlimPajama, two pivotal observations emerged: (1) Global deduplication vs. local deduplication. We analyze and discuss how global (across different sources of datasets) and local (within the single source of dataset) deduplications affect the performance of trained models. (2) Proportions of highly-deduplicated multi-source datasets in the combination. To study this, we construct six configurations on SlimPajama dataset and train individual ones using 1.3B Cerebras-GPT model with Alibi and SwiGLU. Our best configuration outperforms the 1.3B model trained on RedPajama using the same number of training tokens by a significant margin. All our 1.3B models are trained on Cerebras 16$\times$ CS-2 cluster with a total of 80 PFLOP/s in bf16 mixed precision. We further extend our discoveries (such as increasing data diversity is crucial after global deduplication) on a 7B model with large batch-size training. Our SlimPajama-DC models are available at: https://huggingface.co/MBZUAI-LLM/SlimPajama-DC and the separate SlimPajama-DC datasets are available at: https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05758v1">Exploring the Potential of Human-LLM Synergy in Advancing Qualitative Analysis: A Case Study on Mental-Illness Stigma</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 55 pages
    </div>
    <details class="paper-abstract">
      Qualitative analysis is a challenging, yet crucial aspect of advancing research in the field of Human-Computer Interaction (HCI). Recent studies show that large language models (LLMs) can perform qualitative coding within existing schemes, but their potential for collaborative human-LLM discovery and new insight generation in qualitative analysis is still underexplored. To bridge this gap and advance qualitative analysis by harnessing the power of LLMs, we propose CHALET, a novel methodology that leverages the human-LLM collaboration paradigm to facilitate conceptualization and empower qualitative research. The CHALET approach involves LLM-supported data collection, performing both human and LLM deductive coding to identify disagreements, and performing collaborative inductive coding on these disagreement cases to derive new conceptual insights. We validated the effectiveness of CHALET through its application to the attribution model of mental-illness stigma, uncovering implicit stigmatization themes on cognitive, emotional and behavioral dimensions. We discuss the implications for future research, methodology, and the transdisciplinary opportunities CHALET presents for the HCI community and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05999v1">LLMPot: Automated LLM-based Industrial Protocol and Physical Process Emulation for ICS Honeypots</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05610v1">Chain of Attack: a Semantic-Driven Contextual Multi-Turn attacker for LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance in various natural language processing tasks, especially in dialogue systems. However, LLM may also pose security and moral threats, especially in multi round conversations where large models are more easily guided by contextual content, resulting in harmful or biased responses. In this paper, we present a novel method to attack LLMs in multi-turn dialogues, called CoA (Chain of Attack). CoA is a semantic-driven contextual multi-turn attack method that adaptively adjusts the attack policy through contextual feedback and semantic relevance during multi-turn of dialogue with a large model, resulting in the model producing unreasonable or harmful content. We evaluate CoA on different LLMs and datasets, and show that it can effectively expose the vulnerabilities of LLMs, and outperform existing attack methods. Our work provides a new perspective and tool for attacking and defending LLMs, and contributes to the security and ethical assessment of dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06705v1">LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 To appear at IJCAI 2024
    </div>
    <details class="paper-abstract">
      Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05548v1">Investigating Interaction Modes and User Agency in Human-LLM Collaboration for Domain-Specific Data Analysis</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 CHI'24 Late-Breaking Work
    </div>
    <details class="paper-abstract">
      Despite demonstrating robust capabilities in performing tasks related to general-domain data-operation tasks, Large Language Models (LLMs) may exhibit shortcomings when applied to domain-specific tasks. We consider the design of domain-specific AI-powered data analysis tools from two dimensions: interaction and user agency. We implemented two design probes that fall on the two ends of the two dimensions: an open-ended high agency (OHA) prototype and a structured low agency (SLA) prototype. We conducted an interview study with nine data scientists to investigate (1) how users perceived the LLM outputs for data analysis assistance, and (2) how the two test design probes, OHA and SLA, affected user behavior, performance, and perceptions. Our study revealed insights regarding participants' interactions with LLMs, how they perceived the results, and their desire for explainability concerning LLM outputs, along with a noted need for collaboration with other users, and how they envisioned the utility of LLMs in their workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05248v2">LLMs with Personalities in Multi-issue Negotiation Games</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Powered by large language models (LLMs), AI agents have become capable of many human tasks. Using the most canonical definitions of the Big Five personality, we measure the ability of LLMs to negotiate within a game-theoretical framework, as well as methodological challenges to measuring notions of fairness and risk. Simulations (n=1,500) for both single-issue and multi-issue negotiation reveal increase in domain complexity with asymmetric issue valuations improve agreement rates but decrease surplus from aggressive negotiation. Through gradient-boosted regression and Shapley explainers, we find high openness, conscientiousness, and neuroticism are associated with fair tendencies; low agreeableness and low openness are associated with rational tendencies. Low conscientiousness is associated with high toxicity. These results indicate that LLMs may have built-in guardrails that default to fair behavior, but can be "jail broken" to exploit agreeable opponents. We also offer pragmatic insight in how negotiation bots can be designed, and a framework of assessing negotiation behavior based on game theory and computational social science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02228v2">REASONS: A benchmark for REtrieval and Automated citationS Of scieNtific Sentences using Public and Proprietary LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-09
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Automatic citation generation for sentences in a document or report is paramount for intelligence analysts, cybersecurity, news agencies, and education personnel. In this research, we investigate whether large language models (LLMs) are capable of generating references based on two forms of sentence queries: (a) Direct Queries, LLMs are asked to provide author names of the given research article, and (b) Indirect Queries, LLMs are asked to provide the title of a mentioned article when given a sentence from a different article. To demonstrate where LLM stands in this task, we introduce a large dataset called REASONS comprising abstracts of the 12 most popular domains of scientific research on arXiv. From around 20K research articles, we make the following deductions on public and proprietary LLMs: (a) State-of-the-art, often called anthropomorphic GPT-4 and GPT-3.5, suffers from high pass percentage (PP) to minimize the hallucination rate (HR). When tested with Perplexity.ai (7B), they unexpectedly made more errors; (b) Augmenting relevant metadata lowered the PP and gave the lowest HR; (c) Advance retrieval-augmented generation (RAG) using Mistral demonstrates consistent and robust citation support on indirect queries and matched performance to GPT-3.5 and GPT-4. The HR across all domains and models decreased by an average of 41.93%, and the PP was reduced to 0% in most cases. In terms of generation quality, the average F1 Score and BLEU were 68.09% and 57.51%, respectively; (d) Testing with adversarial samples showed that LLMs, including the Advance RAG Mistral, struggle to understand context, but the extent of this issue was small in Mistral and GPT-4-Preview. Our study contributes valuable insights into the reliability of RAG for automated citation generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05469v1">PLLM-CS: Pre-trained Large Language Model (LLM) for Cyber Threat Detection in Satellite Networks</a></div>
    <div class="paper-meta">
      📅 2024-05-09
    </div>
    <details class="paper-abstract">
      Satellite networks are vital in facilitating communication services for various critical infrastructures. These networks can seamlessly integrate with a diverse array of systems. However, some of these systems are vulnerable due to the absence of effective intrusion detection systems, which can be attributed to limited research and the high costs associated with deploying, fine-tuning, monitoring, and responding to security breaches. To address these challenges, we propose a pretrained Large Language Model for Cyber Security , for short PLLM-CS, which is a variant of pre-trained Transformers [1], which includes a specialized module for transforming network data into contextually suitable inputs. This transformation enables the proposed LLM to encode contextual information within the cyber data. To validate the efficacy of the proposed method, we conducted empirical experiments using two publicly available network datasets, UNSW_NB 15 and TON_IoT, both providing Internet of Things (IoT)-based traffic data. Our experiments demonstrate that proposed LLM method outperforms state-of-the-art techniques such as BiLSTM, GRU, and CNN. Notably, the PLLM-CS method achieves an outstanding accuracy level of 100% on the UNSW_NB 15 dataset, setting a new standard for benchmark performance in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05444v1">Evaluating Students' Open-ended Written Responses with LLMs: Using the RAG Framework for GPT-3.5, GPT-4, Claude-3, and Mistral-Large</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 18 pages, 6 tables, 1 figure
    </div>
    <details class="paper-abstract">
      Evaluating open-ended written examination responses from students is an essential yet time-intensive task for educators, requiring a high degree of effort, consistency, and precision. Recent developments in Large Language Models (LLMs) present a promising opportunity to balance the need for thorough evaluation with efficient use of educators' time. In our study, we explore the effectiveness of LLMs ChatGPT-3.5, ChatGPT-4, Claude-3, and Mistral-Large in assessing university students' open-ended answers to questions made about reference material they have studied. Each model was instructed to evaluate 54 answers repeatedly under two conditions: 10 times (10-shot) with a temperature setting of 0.0 and 10 times with a temperature of 0.5, expecting a total of 1,080 evaluations per model and 4,320 evaluations across all models. The RAG (Retrieval Augmented Generation) framework was used as the framework to make the LLMs to process the evaluation of the answers. As of spring 2024, our analysis revealed notable variations in consistency and the grading outcomes provided by studied LLMs. There is a need to comprehend strengths and weaknesses of LLMs in educational settings for evaluating open-ended written responses. Further comparative research is essential to determine the accuracy and cost-effectiveness of using LLMs for educational assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06703v1">Interpretable Cross-Examination Technique (ICE-T): Using highly informative features to boost LLM performance</a></div>
    <div class="paper-meta">
      📅 2024-05-08
    </div>
    <details class="paper-abstract">
      In this paper, we introduce the Interpretable Cross-Examination Technique (ICE-T), a novel approach that leverages structured multi-prompt techniques with Large Language Models (LLMs) to improve classification performance over zero-shot and few-shot methods. In domains where interpretability is crucial, such as medicine and law, standard models often fall short due to their "black-box" nature. ICE-T addresses these limitations by using a series of generated prompts that allow an LLM to approach the problem from multiple directions. The responses from the LLM are then converted into numerical feature vectors and processed by a traditional classifier. This method not only maintains high interpretability but also allows for smaller, less capable models to achieve or exceed the performance of larger, more advanced models under zero-shot conditions. We demonstrate the effectiveness of ICE-T across a diverse set of data sources, including medical records and legal documents, consistently surpassing the zero-shot baseline in terms of classification metrics such as F1 scores. Our results indicate that ICE-T can be used for improving both the performance and transparency of AI applications in complex decision-making environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05378v1">"They are uncultured": Unveiling Covert Harms and Social Threats in LLM Generated Conversations</a></div>
    <div class="paper-meta">
      📅 2024-05-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as an integral part of modern societies, powering user-facing applications such as personal assistants and enterprise applications like recruitment tools. Despite their utility, research indicates that LLMs perpetuate systemic biases. Yet, prior works on LLM harms predominantly focus on Western concepts like race and gender, often overlooking cultural concepts from other parts of the world. Additionally, these studies typically investigate "harm" as a singular dimension, ignoring the various and subtle forms in which harms manifest. To address this gap, we introduce the Covert Harms and Social Threats (CHAST), a set of seven metrics grounded in social science literature. We utilize evaluation models aligned with human assessments to examine the presence of covert harms in LLM-generated conversations, particularly in the context of recruitment. Our experiments reveal that seven out of the eight LLMs included in this study generated conversations riddled with CHAST, characterized by malign views expressed in seemingly neutral language unlikely to be detected by existing methods. Notably, these LLMs manifested more extreme views and opinions when dealing with non-Western concepts like caste, compared to Western ones such as race.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05348v1">The Effect of Model Size on LLM Post-hoc Explainability via LIME</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 Published at ICLR 2024 Workshop on Secure and Trustworthy Large Language Models
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming bigger to boost performance. However, little is known about how explainability is affected by this trend. This work explores LIME explanations for DeBERTaV3 models of four different sizes on natural language inference (NLI) and zero-shot classification (ZSC) tasks. We evaluate the explanations based on their faithfulness to the models' internal decision processes and their plausibility, i.e. their agreement with human explanations. The key finding is that increased model size does not correlate with plausibility despite improved model performance, suggesting a misalignment between the LIME explanations and the models' internal processes as model size increases. Our results further suggest limitations regarding faithfulness metrics in NLI contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05345v1">QuaLLM: An LLM-based Framework to Extract Quantitative Insights from Online Forums</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 Accepted to CHI LLM as Research Tools Workshop (2024)
    </div>
    <details class="paper-abstract">
      Online discussion forums provide crucial data to understand the concerns of a wide range of real-world communities. However, the typical qualitative and quantitative methods used to analyze those data, such as thematic analysis and topic modeling, are infeasible to scale or require significant human effort to translate outputs to human readable forms. This study introduces QuaLLM, a novel LLM-based framework to analyze and extract quantitative insights from text data on online forums. The framework consists of a novel prompting methodology and evaluation strategy. We applied this framework to analyze over one million comments from two Reddit's rideshare worker communities, marking the largest study of its type. We uncover significant worker concerns regarding AI and algorithmic platform decisions, responding to regulatory calls about worker insights. In short, our work sets a new precedent for AI-assisted quantitative data analysis to surface concerns from online forums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05253v1">Open Source Language Models Can Provide Feedback: Evaluating LLMs' Ability to Help Students Using GPT-4-As-A-Judge</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 7 pages, 4 figures, 2 tables. Accepted for publication at the 29th annual ACM conference on Innovation and Technology in Computer Science Education (ITiCSE 2024)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great potential for the automatic generation of feedback in a wide range of computing contexts. However, concerns have been voiced around the privacy and ethical implications of sending student work to proprietary models. This has sparked considerable interest in the use of open source LLMs in education, but the quality of the feedback that such open models can produce remains understudied. This is a concern as providing flawed or misleading generated feedback could be detrimental to student learning. Inspired by recent work that has utilised very powerful LLMs, such as GPT-4, to evaluate the outputs produced by less powerful models, we conduct an automated analysis of the quality of the feedback produced by several open source models using a dataset from an introductory programming course. First, we investigate the viability of employing GPT-4 as an automated evaluator by comparing its evaluations with those of a human expert. We observe that GPT-4 demonstrates a bias toward positively rating feedback while exhibiting moderate agreement with human raters, showcasing its potential as a feedback evaluator. Second, we explore the quality of feedback generated by several leading open-source LLMs by using GPT-4 to evaluate the feedback. We find that some models offer competitive performance with popular proprietary LLMs, such as ChatGPT, indicating opportunities for their responsible use in educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15667v4">The Promise and Challenges of Using LLMs to Accelerate the Screening Process of Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 Accepted to the International Conference on Evaluation and Assessment in Software Engineering (EASE), 2024 edition
    </div>
    <details class="paper-abstract">
      Systematic review (SR) is a popular research method in software engineering (SE). However, conducting an SR takes an average of 67 weeks. Thus, automating any step of the SR process could reduce the effort associated with SRs. Our objective is to investigate if Large Language Models (LLMs) can accelerate title-abstract screening by simplifying abstracts for human screeners, and automating title-abstract screening. We performed an experiment where humans screened titles and abstracts for 20 papers with both original and simplified abstracts from a prior SR. The experiment with human screeners was reproduced with GPT-3.5 and GPT-4 LLMs to perform the same screening tasks. We also studied if different prompting techniques (Zero-shot (ZS), One-shot (OS), Few-shot (FS), and Few-shot with Chain-of-Thought (FS-CoT)) improve the screening performance of LLMs. Lastly, we studied if redesigning the prompt used in the LLM reproduction of screening leads to improved performance. Text simplification did not increase the screeners' screening performance, but reduced the time used in screening. Screeners' scientific literacy skills and researcher status predict screening performance. Some LLM and prompt combinations perform as well as human screeners in the screening tasks. Our results indicate that the GPT-4 LLM is better than its predecessor, GPT-3.5. Additionally, Few-shot and One-shot prompting outperforms Zero-shot prompting. Using LLMs for text simplification in the screening process does not significantly improve human performance. Using LLMs to automate title-abstract screening seems promising, but current LLMs are not significantly more accurate than human screeners. To recommend the use of LLMs in the screening process of SRs, more research is needed. We recommend future SR studies publish replication packages with screening data to enable more conclusive experimenting with LLM screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06700v1">LLM-Augmented Agent-Based Modelling for Social Simulations: Challenges and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 11 pages, 0 figures, Hybrid Human Artificial Intelligence (HHAI) 2024
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to make significant strides, their better integration into agent-based simulations offers a transformational potential for understanding complex social systems. However, such integration is not trivial and poses numerous challenges. Based on this observation, in this paper, we explore architectures and methods to systematically develop LLM-augmented social simulations and discuss potential research directions in this field. We conclude that integrating LLMs with agent-based simulations offers a powerful toolset for researchers and scientists, allowing for more nuanced, realistic, and comprehensive models of complex systems and human behaviours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05459v2">Personal LLM Agents: Insights and Survey about the Capability, Efficiency and Security</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 https://github.com/MobileLLM/Personal_LLM_Agents_Survey
    </div>
    <details class="paper-abstract">
      Since the advent of personal computing devices, intelligent personal assistants (IPAs) have been one of the key technologies that researchers and engineers have focused on, aiming to help users efficiently obtain information and execute tasks, and provide users with more intelligent, convenient, and rich interaction experiences. With the development of smartphones and IoT, computing and sensing devices have become ubiquitous, greatly expanding the boundaries of IPAs. However, due to the lack of capabilities such as user intent understanding, task planning, tool using, and personal data management etc., existing IPAs still have limited practicality and scalability. Recently, the emergence of foundation models, represented by large language models (LLMs), brings new opportunities for the development of IPAs. With the powerful semantic understanding and reasoning capabilities, LLM can enable intelligent agents to solve complex problems autonomously. In this paper, we focus on Personal LLM Agents, which are LLM-based agents that are deeply integrated with personal data and personal devices and used for personal assistance. We envision that Personal LLM Agents will become a major software paradigm for end-users in the upcoming era. To realize this vision, we take the first step to discuss several important questions about Personal LLM Agents, including their architecture, capability, efficiency and security. We start by summarizing the key components and design choices in the architecture of Personal LLM Agents, followed by an in-depth analysis of the opinions collected from domain experts. Next, we discuss several key challenges to achieve intelligent, efficient and secure Personal LLM Agents, followed by a comprehensive survey of representative solutions to address these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06221v2">ResumeFlow: An LLM-facilitated Pipeline for Personalized Resume Generation and Refinement</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 Accepted to SIGIR 2024 (Demo)
    </div>
    <details class="paper-abstract">
      Crafting the ideal, job-specific resume is a challenging task for many job applicants, especially for early-career applicants. While it is highly recommended that applicants tailor their resume to the specific role they are applying for, manually tailoring resumes to job descriptions and role-specific requirements is often (1) extremely time-consuming, and (2) prone to human errors. Furthermore, performing such a tailoring step at scale while applying to several roles may result in a lack of quality of the edited resumes. To tackle this problem, in this demo paper, we propose ResumeFlow: a Large Language Model (LLM) aided tool that enables an end user to simply provide their detailed resume and the desired job posting, and obtain a personalized resume specifically tailored to that specific job posting in the matter of a few seconds. Our proposed pipeline leverages the language understanding and information extraction capabilities of state-of-the-art LLMs such as OpenAI's GPT-4 and Google's Gemini, in order to (1) extract details from a job description, (2) extract role-specific details from the user-provided resume, and then (3) use these to refine and generate a role-specific resume for the user. Our easy-to-use tool leverages the user-chosen LLM in a completely off-the-shelf manner, thus requiring no fine-tuning. We demonstrate the effectiveness of our tool via a video demo and propose novel task-specific evaluation metrics to control for alignment and hallucination. Our tool is available at https://job-aligned-resume.streamlit.app.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03146v2">Quantifying the Capabilities of LLMs across Scale and Precision</a></div>
    <div class="paper-meta">
      📅 2024-05-08
    </div>
    <details class="paper-abstract">
      Scale is often attributed as one of the factors that cause an increase in the performance of LLMs, resulting in models with billion and trillion parameters. One of the limitations of such large models is the high computational requirements that limit their usage, deployment, and debugging in resource-constrained scenarios. Two commonly used alternatives to bypass these limitations are to use the smaller versions of LLMs (e.g. Llama 7B instead of Llama 70B) and lower the memory requirements by using quantization. While these approaches effectively address the limitation of resources, their impact on model performance needs thorough examination. In this study, we perform a comprehensive evaluation to investigate the effect of model scale and quantization on the performance. We experiment with two major families of open-source instruct models ranging from 7 billion to 70 billion parameters. Our extensive zero-shot experiments across various tasks including natural language understanding, reasoning, misinformation detection, and hallucination reveal that larger models generally outperform their smaller counterparts, suggesting that scale remains an important factor in enhancing performance. We found that larger models show exceptional resilience to precision reduction and can maintain high accuracy even at 4-bit quantization for numerous tasks and they serve as a better solution than using smaller models at high precision under similar memory requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04727v1">LLMs Can Patch Up Missing Relevance Judgments in Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-05-08
      | 💬 5 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Unjudged documents or holes in information retrieval benchmarks are considered non-relevant in evaluation, yielding no gains in measuring effectiveness. However, these missing judgments may inadvertently introduce biases into the evaluation as their prevalence for a retrieval model is heavily contingent on the pooling process. Thus, filling holes becomes crucial in ensuring reliable and accurate evaluation. Collecting human judgment for all documents is cumbersome and impractical. In this paper, we aim at leveraging large language models (LLMs) to automatically label unjudged documents. Our goal is to instruct an LLM using detailed instructions to assign fine-grained relevance judgments to holes. To this end, we systematically simulate scenarios with varying degrees of holes by randomly dropping relevant documents from the relevance judgment in TREC DL tracks. Our experiments reveal a strong correlation between our LLM-based method and ground-truth relevance judgments. Based on our simulation experiments conducted on three TREC DL datasets, in the extreme scenario of retaining only 10% of judgments, our method achieves a Kendall tau correlation of 0.87 and 0.92 on an average for Vicu\~na-7B and GPT-3.5 Turbo respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04656v1">Corporate Communication Companion (CCC): An LLM-empowered Writing Assistant for Workplace Social Media</a></div>
    <div class="paper-meta">
      📅 2024-05-07
    </div>
    <details class="paper-abstract">
      Workplace social media platforms enable employees to cultivate their professional image and connect with colleagues in a semi-formal environment. While semi-formal corporate communication poses a unique set of challenges, large language models (LLMs) have shown great promise in helping users draft and edit their social media posts. However, LLMs may fail to capture individualized tones and voices in such workplace use cases, as they often generate text using a "one-size-fits-all" approach that can be perceived as generic and bland. In this paper, we present Corporate Communication Companion (CCC), an LLM-empowered interactive system that helps people compose customized and individualized workplace social media posts. Using need-finding interviews to motivate our system design, CCC decomposes the writing process into two core functions, outline and edit: First, it suggests post outlines based on users' job status and previous posts, and next provides edits with attributions that users can contextually customize. We conducted a within-subjects user study asking participants both to write posts and evaluate posts written by others. The results show that CCC enhances users' writing experience, and audience members rate CCC-enhanced posts as higher quality than posts written using a non-customized writing assistant. We conclude by discussing the implications of LLM-empowered corporate communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17438v1">An LLM-Tool Compiler for Fused Parallel Function Calling</a></div>
    <div class="paper-meta">
      📅 2024-05-07
    </div>
    <details class="paper-abstract">
      State-of-the-art sequential reasoning in Large Language Models (LLMs) has expanded the capabilities of Copilots beyond conversational tasks to complex function calling, managing thousands of API calls. However, the tendency of compositional prompting to segment tasks into multiple steps, each requiring a round-trip to the GPT APIs, leads to increased system latency and costs. Although recent advancements in parallel function calling have improved tool execution per API call, they may necessitate more detailed in-context instructions and task breakdown at the prompt level, resulting in higher engineering and production costs. Inspired by the hardware design principles of multiply-add (MAD) operations, which fuse multiple arithmetic operations into a single task from the compiler's perspective, we propose LLM-Tool Compiler, which selectively fuses similar types of tool operations under a single function at runtime, presenting them as a unified task to the LLM. This selective fusion inherently enhances parallelization and efficiency. Benchmarked on a large-scale Copilot platform, LLM-Tool Compiler achieves up to four times more parallel calls than existing methods, reducing token costs and latency by up to 40% and 12%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04403v1">Learning To See But Forgetting To Follow: Visual Instruction Tuning Makes LLMs More Prone To Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2024-05-07
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) with image-understanding capabilities has resulted in a boom of high-performing Vision-Language models (VLMs). While studying the alignment of LLMs to human values has received widespread attention, the safety of VLMs has not received the same attention. In this paper, we explore the impact of jailbreaking on three state-of-the-art VLMs, each using a distinct modeling approach. By comparing each VLM to their respective LLM backbone, we find that each VLM is more susceptible to jailbreaking. We consider this as an undesirable outcome from visual instruction-tuning, which imposes a forgetting effect on an LLM's safety guardrails. Therefore, we provide recommendations for future work based on evaluation strategies that aim to highlight the weaknesses of a VLM, as well as take safety measures into account during visual instruction tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04286v1">Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore</a></div>
    <div class="paper-meta">
      📅 2024-05-07
    </div>
    <details class="paper-abstract">
      The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04282v1">CoqPyt: Proof Navigation in Python in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-07
      | 💬 Accepted to FSE '24 Demonstrations Track
    </div>
    <details class="paper-abstract">
      Proof assistants enable users to develop machine-checked proofs regarding software-related properties. Unfortunately, the interactive nature of these proof assistants imposes most of the proof burden on the user, making formal verification a complex, and time-consuming endeavor. Recent automation techniques based on neural methods address this issue, but require good programmatic support for collecting data and interacting with proof assistants. This paper presents CoqPyt, a Python tool for interacting with the Coq proof assistant. CoqPyt improves on other Coq-related tools by providing novel features, such as the extraction of rich premise data. We expect our work to aid development of tools and techniques, especially LLM-based, designed for proof synthesis and repair. A video describing and demonstrating CoqPyt is available at: https://youtu.be/fk74o0rePM8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.04215v1">NL2Plan: Robust LLM-Driven Planning from Minimal Text Descriptions</a></div>
    <div class="paper-meta">
      📅 2024-05-07
      | 💬 Accepted for the ICAPS 2024 Workshop on Human-Aware and Explainable Planning
    </div>
    <details class="paper-abstract">
      Today's classical planners are powerful, but modeling input tasks in formats such as PDDL is tedious and error-prone. In contrast, planning with Large Language Models (LLMs) allows for almost any input text, but offers no guarantees on plan quality or even soundness. In an attempt to merge the best of these two approaches, some work has begun to use LLMs to automate parts of the PDDL creation process. However, these methods still require various degrees of expert input. We present NL2Plan, the first domain-agnostic offline LLM-driven planning system. NL2Plan uses an LLM to incrementally extract the necessary information from a short text prompt before creating a complete PDDL description of both the domain and the problem, which is finally solved by a classical planner. We evaluate NL2Plan on four planning domains and find that it solves 10 out of 15 tasks - a clear improvement over a plain chain-of-thought reasoning LLM approach, which only solves 2 tasks. Moreover, in two out of the five failure cases, instead of returning an invalid plan, NL2Plan reports that it failed to solve the task. In addition to using NL2Plan in end-to-end mode, users can inspect and correct all of its intermediate results, such as the PDDL representation, increasing explainability and making it an assistive tool for PDDL creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03654v2">Can LLMs Deeply Detect Complex Malicious Queries? A Framework for Jailbreaking via Obfuscating Intent</a></div>
    <div class="paper-meta">
      📅 2024-05-07
    </div>
    <details class="paper-abstract">
      To demonstrate and address the underlying maliciousness, we propose a theoretical hypothesis and analytical approach, and introduce a new black-box jailbreak attack methodology named IntentObfuscator, exploiting this identified flaw by obfuscating the true intentions behind user prompts.This approach compels LLMs to inadvertently generate restricted content, bypassing their built-in content security measures. We detail two implementations under this framework: "Obscure Intention" and "Create Ambiguity", which manipulate query complexity and ambiguity to evade malicious intent detection effectively. We empirically validate the effectiveness of the IntentObfuscator method across several models, including ChatGPT-3.5, ChatGPT-4, Qwen and Baichuan, achieving an average jailbreak success rate of 69.21\%. Notably, our tests on ChatGPT-3.5, which claims 100 million weekly active users, achieved a remarkable success rate of 83.65\%. We also extend our validation to diverse types of sensitive content like graphic violence, racism, sexism, political sensitivity, cybersecurity threats, and criminal skills, further proving the substantial impact of our findings on enhancing 'Red Team' strategies against LLM content security frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.04764v2">ChatUniTest: A Framework for LLM-Based Test Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-07
      | 💬 This shorter version is accepted by the FSE 2024 Demonstrations Track, and the previous longer version titled "ChatUniTest: a ChatGPT-based automated unit test generation tool" can be found at arXiv:2305.04764v1
    </div>
    <details class="paper-abstract">
      Unit testing is an essential yet frequently arduous task. Various automated unit test generation tools have been introduced to mitigate this challenge. Notably, methods based on large language models (LLMs) have garnered considerable attention and exhibited promising results in recent years. Nevertheless, LLM-based tools encounter limitations in generating accurate unit tests. This paper presents ChatUniTest, an LLM-based automated unit test generation framework. ChatUniTest incorporates an adaptive focal context mechanism to encompass valuable context in prompts and adheres to a generation-validation-repair mechanism to rectify errors in generated unit tests. Subsequently, we have developed ChatUniTest Core, a common library that implements core workflow, complemented by the ChatUniTest Toolchain, a suite of seamlessly integrated tools enhancing the capabilities of ChatUniTest. Our effectiveness evaluation reveals that ChatUniTest outperforms TestSpark and EvoSuite in half of the evaluated projects, achieving the highest overall line coverage. Furthermore, insights from our user study affirm that ChatUniTest delivers substantial value to various stakeholders in the software testing domain. ChatUniTest is available at https://github.com/ZJU-ACES-ISE/ChatUniTest, and the demo video is available at https://www.youtube.com/watch?v=GmfxQUqm2ZQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03901v1">OmniActions: Predicting Digital Actions in Response to Real-World Multimodal Sensory Inputs with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 Paper accepted to the 2024 CHI Conference on Human Factors in Computing Systems (CHI 2024)
    </div>
    <details class="paper-abstract">
      The progression to "Pervasive Augmented Reality" envisions easy access to multimodal information continuously. However, in many everyday scenarios, users are occupied physically, cognitively or socially. This may increase the friction to act upon the multimodal information that users encounter in the world. To reduce such friction, future interactive interfaces should intelligently provide quick access to digital actions based on users' context. To explore the range of possible digital actions, we conducted a diary study that required participants to capture and share the media that they intended to perform actions on (e.g., images or audio), along with their desired actions and other contextual information. Using this data, we generated a holistic design space of digital follow-up actions that could be performed in response to different types of multimodal sensory inputs. We then designed OmniActions, a pipeline powered by large language models (LLMs) that processes multimodal sensory inputs and predicts follow-up actions on the target information grounded in the derived design space. Using the empirical data collected in the diary study, we performed quantitative evaluations on three variations of LLM techniques (intent classification, in-context learning and finetuning) and identified the most effective technique for our task. Additionally, as an instantiation of the pipeline, we developed an interactive prototype and reported preliminary user feedback about how people perceive and react to the action predictions and its errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12545v2">TrustScore: Reference-Free Evaluation of LLM Response Trustworthiness</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, prompting a surge in their practical applications. However, concerns have arisen regarding the trustworthiness of LLMs outputs, particularly in closed-book question-answering tasks, where non-experts may struggle to identify inaccuracies due to the absence of contextual or ground truth information. This paper introduces TrustScore, a framework based on the concept of Behavioral Consistency, which evaluates whether an LLMs response aligns with its intrinsic knowledge. Additionally, TrustScore can seamlessly integrate with fact-checking methods, which assesses alignment with external knowledge sources. The experimental results show that TrustScore achieves strong correlations with human judgments, surpassing existing reference-free metrics, and achieving results on par with reference-based metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03845v1">Self-Improving Customer Review Response Generation Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 18 pages, 4 figure, 8 figures in Appendix, accepted to LREC-COLING 2024 workshop
    </div>
    <details class="paper-abstract">
      Previous studies have demonstrated that proactive interaction with user reviews has a positive impact on the perception of app users and encourages them to submit revised ratings. Nevertheless, developers encounter challenges in managing a high volume of reviews, particularly in the case of popular apps with a substantial influx of daily reviews. Consequently, there is a demand for automated solutions aimed at streamlining the process of responding to user reviews. To address this, we have developed a new system for generating automatic responses by leveraging user-contributed documents with the help of retrieval-augmented generation (RAG) and advanced Large Language Models (LLMs). Our solution, named SCRABLE, represents an adaptive customer review response automation that enhances itself with self-optimizing prompts and a judging mechanism based on LLMs. Additionally, we introduce an automatic scoring mechanism that mimics the role of a human evaluator to assess the quality of responses generated in customer review domains. Extensive experiments and analyses conducted on real-world datasets reveal that our method is effective in producing high-quality responses, yielding improvement of more than 8.5% compared to the baseline. Further validation through manual examination of the generated responses underscores the efficacy our proposed system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19705v2">When to Retrieve: Teaching LLMs to Utilize Information Retrieval Effectively</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      In this paper, we demonstrate how Large Language Models (LLMs) can effectively learn to use an off-the-shelf information retrieval (IR) system specifically when additional context is required to answer a given question. Given the performance of IR systems, the optimal strategy for question answering does not always entail external information retrieval; rather, it often involves leveraging the parametric memory of the LLM itself. Prior research has identified this phenomenon in the PopQA dataset, wherein the most popular questions are effectively addressed using the LLM's parametric memory, while less popular ones require IR system usage. Following this, we propose a tailored training approach for LLMs, leveraging existing open-domain question answering datasets. Here, LLMs are trained to generate a special token, <RET>, when they do not know the answer to a question. Our evaluation of the Adaptive Retrieval LLM (Adapt-LLM) on the PopQA dataset showcases improvements over the same LLM under three configurations: (i) retrieving information for all the questions, (ii) using always the parametric memory of the LLM, and (iii) using a popularity threshold to decide when to use a retriever. Through our analysis, we demonstrate that Adapt-LLM is able to generate the <RET> token when it determines that it does not know how to answer a question, indicating the need for IR, while it achieves notably high accuracy levels when it chooses to rely only on its parametric memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03677v1">Towards A Human-in-the-Loop LLM Approach to Collaborative Discourse Analysis</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 In press at the 25th international conference on Artificial Intelligence in Education (AIED) Late-Breaking Results (LBR) track
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated proficiency in contextualizing their outputs using human input, often matching or beating human-level performance on a variety of tasks. However, LLMs have not yet been used to characterize synergistic learning in students' collaborative discourse. In this exploratory work, we take a first step towards adopting a human-in-the-loop prompt engineering approach with GPT-4-Turbo to summarize and categorize students' synergistic learning during collaborative discourse. Our preliminary findings suggest GPT-4-Turbo may be able to characterize students' synergistic learning in a manner comparable to humans and that our approach warrants further investigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00801v2">"Ask Me Anything": How Comcast Uses LLMs to Assist Agents in Real Time</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Customer service is how companies interface with their customers. It can contribute heavily towards the overall customer satisfaction. However, high-quality service can become expensive, creating an incentive to make it as cost efficient as possible and prompting most companies to utilize AI-powered assistants, or "chat bots". On the other hand, human-to-human interaction is still desired by customers, especially when it comes to complex scenarios such as disputes and sensitive topics like bill payment. This raises the bar for customer service agents. They need to accurately understand the customer's question or concern, identify a solution that is acceptable yet feasible (and within the company's policy), all while handling multiple conversations at once. In this work, we introduce "Ask Me Anything" (AMA) as an add-on feature to an agent-facing customer service interface. AMA allows agents to ask questions to a large language model (LLM) on demand, as they are handling customer conversations -- the LLM provides accurate responses in real-time, reducing the amount of context switching the agent needs. In our internal experiments, we find that agents using AMA versus a traditional search experience spend approximately 10% fewer seconds per conversation containing a search, translating to millions of dollars of savings annually. Agents that used the AMA feature provided positive feedback nearly 80% of the time, demonstrating its usefulness as an AI-assisted feature for customer care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20288v2">Can LLMs Correct Physicians, Yet? Investigating Effective Interaction Methods in the Medical Domain</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 Accepted for oral presentation at NAACL 2024, The 6th Clinical Natural Language Processing Workshop
    </div>
    <details class="paper-abstract">
      We explore the potential of Large Language Models (LLMs) to assist and potentially correct physicians in medical decision-making tasks. We evaluate several LLMs, including Meditron, Llama2, and Mistral, to analyze the ability of these models to interact effectively with physicians across different scenarios. We consider questions from PubMedQA and several tasks, ranging from binary (yes/no) responses to long answer generation, where the answer of the model is produced after an interaction with a physician. Our findings suggest that prompt design significantly influences the downstream accuracy of LLMs and that LLMs can provide valuable feedback to physicians, challenging incorrect diagnoses and contributing to more accurate decision-making. For example, when the physician is accurate 38% of the time, Mistral can produce the correct answer, improving accuracy up to 74% depending on the prompt being used, while Llama2 and Meditron models exhibit greater sensitivity to prompt choice. Our analysis also uncovers the challenges of ensuring that LLM-generated suggestions are pertinent and useful, emphasizing the need for further research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03480v1">Doing Personal LAPS: LLM-Augmented Dialogue Construction for Personalized Multi-Session Conversational Search</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 Accepted at SIGIR 2024 (Full Paper)
    </div>
    <details class="paper-abstract">
      The future of conversational agents will provide users with personalized information responses. However, a significant challenge in developing models is the lack of large-scale dialogue datasets that span multiple sessions and reflect real-world user preferences. Previous approaches rely on experts in a wizard-of-oz setup that is difficult to scale, particularly for personalized tasks. Our method, LAPS, addresses this by using large language models (LLMs) to guide a single human worker in generating personalized dialogues. This method has proven to speed up the creation process and improve quality. LAPS can collect large-scale, human-written, multi-session, and multi-domain conversations, including extracting user preferences. When compared to existing datasets, LAPS-produced conversations are as natural and diverse as expert-created ones, which stays in contrast with fully synthetic methods. The collected dataset is suited to train preference extraction and personalized response generation. Our results show that responses generated explicitly using extracted preferences better match user's actual preferences, highlighting the value of using extracted preferences over simple dialogue history. Overall, LAPS introduces a new method to leverage LLMs to create realistic personalized conversational data more efficiently and effectively than previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00595v3">State of What Art? A Call for Multi-Prompt LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 Accepted at TACL; pre-MIT Press publication version
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to the development of various evaluation benchmarks. These benchmarks typically rely on a single instruction template for evaluating all LLMs on a specific task. In this paper, we comprehensively analyze the brittleness of results obtained via single-prompt evaluations across 6.5M instances, involving 20 different LLMs and 39 tasks from 3 benchmarks. To improve robustness of the analysis, we propose to evaluate LLMs with a set of diverse prompts instead. We discuss tailored evaluation metrics for specific use cases (e.g., LLM developers vs. developers interested in a specific downstream task), ensuring a more reliable and meaningful assessment of LLM capabilities. We then implement these criteria and conduct evaluations of multiple models, providing insights into the true strengths and limitations of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05175v3">Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), renowned for their remarkable performance across diverse domains, present a challenge when it comes to practical deployment due to their colossal model size. In response to this challenge, efforts have been directed toward the application of traditional network pruning techniques to LLMs, uncovering a massive number of parameters that can be pruned in one-shot without hurting performance. Prevailing LLM pruning strategies have consistently adhered to the practice of uniformly pruning all layers at equivalent sparsity, resulting in robust performance. However, this observation stands in contrast to the prevailing trends observed in the field of vision models, where non-uniform layerwise sparsity typically yields stronger results. To understand the underlying reasons for this disparity, we conduct a comprehensive study and discover a strong correlation with the emergence of activation outliers in LLMs. Inspired by this finding, we introduce a novel LLM pruning methodology that incorporates a tailored set of non-uniform layerwise sparsity ratios, termed as Outlier Weighed Layerwise sparsity (OWL). The sparsity ratio of OWL is proportional to the outlier ratio observed within each layer, facilitating a more effective alignment between layerwise weight sparsity and outlier ratios. Our empirical evaluation, conducted across the LLaMA-V1 family and OPT, spanning various benchmarks, demonstrates the distinct advantages offered by OWL over previous methods. For instance, OWL exhibits a remarkable performance gain, surpassing the state-of-the-art Wanda and SparseGPT by 61.22 and 6.80 perplexity at a high sparsity level of 70%, respectively, while delivering 2.6x end-to-end inference speed-up in the DeepSparse inference engine. Codes are available at https://github.com/luuyin/OWL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03153v1">Exploring the Potential of the Large Language Models (LLMs) in Identifying Misleading News Headlines</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 5 pages, 2 tables, 1st HEAL Workshop at CHI Conference on Human Factors in Computing Systems, May 12, Honolulu, HI, USA 2024
    </div>
    <details class="paper-abstract">
      In the digital age, the prevalence of misleading news headlines poses a significant challenge to information integrity, necessitating robust detection mechanisms. This study explores the efficacy of Large Language Models (LLMs) in identifying misleading versus non-misleading news headlines. Utilizing a dataset of 60 articles, sourced from both reputable and questionable outlets across health, science & tech, and business domains, we employ three LLMs- ChatGPT-3.5, ChatGPT-4, and Gemini-for classification. Our analysis reveals significant variance in model performance, with ChatGPT-4 demonstrating superior accuracy, especially in cases with unanimous annotator agreement on misleading headlines. The study emphasizes the importance of human-centered evaluation in developing LLMs that can navigate the complexities of misinformation detection, aligning technical proficiency with nuanced human judgment. Our findings contribute to the discourse on AI ethics, emphasizing the need for models that are not only technically advanced but also ethically aligned and sensitive to the subtleties of human interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03152v1">MMGER: Multi-modal and Multi-granularity Generative Error Correction with LLM for Joint Accent and Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2024-05-06
    </div>
    <details class="paper-abstract">
      Despite notable advancements in automatic speech recognition (ASR), performance tends to degrade when faced with adverse conditions. Generative error correction (GER) leverages the exceptional text comprehension capabilities of large language models (LLM), delivering impressive performance in ASR error correction, where N-best hypotheses provide valuable information for transcription prediction. However, GER encounters challenges such as fixed N-best hypotheses, insufficient utilization of acoustic information, and limited specificity to multi-accent scenarios. In this paper, we explore the application of GER in multi-accent scenarios. Accents represent deviations from standard pronunciation norms, and the multi-task learning framework for simultaneous ASR and accent recognition (AR) has effectively addressed the multi-accent scenarios, making it a prominent solution. In this work, we propose a unified ASR-AR GER model, named MMGER, leveraging multi-modal correction, and multi-granularity correction. Multi-task ASR-AR learning is employed to provide dynamic 1-best hypotheses and accent embeddings. Multi-modal correction accomplishes fine-grained frame-level correction by force-aligning the acoustic features of speech with the corresponding character-level 1-best hypothesis sequence. Multi-granularity correction supplements the global linguistic information by incorporating regular 1-best hypotheses atop fine-grained multi-modal correction to achieve coarse-grained utterance-level correction. MMGER effectively mitigates the limitations of GER and tailors LLM-based ASR error correction for the multi-accent scenarios. Experiments conducted on the multi-accent Mandarin KeSpeech dataset demonstrate the efficacy of MMGER, achieving a 26.72% relative improvement in AR accuracy and a 27.55% relative reduction in ASR character error rate, compared to a well-established standard baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14385v2">Generative AI Beyond LLMs: System Implications of Multi-Modal Generation</a></div>
    <div class="paper-meta">
      📅 2024-05-06
      | 💬 Published at 2024 IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS)
    </div>
    <details class="paper-abstract">
      As the development of large-scale Generative AI models evolve beyond text (1D) generation to include image (2D) and video (3D) generation, processing spatial and temporal information presents unique challenges to quality, performance, and efficiency. We present the first work towards understanding this new system design space for multi-modal text-to-image (TTI) and text-to-video (TTV) generation models. Current model architecture designs are bifurcated into 2 categories: Diffusion- and Transformer-based models. Our systematic performance characterization on a suite of eight representative TTI/TTV models shows that after state-of-the-art optimization techniques such as Flash Attention are applied, Convolution accounts for up to 44% of execution time for Diffusion-based TTI models, while Linear layers consume up to 49% of execution time for Transformer-based models. We additionally observe that Diffusion-based TTI models resemble the Prefill stage of LLM inference, and benefit from 1.1-2.5x greater speedup from Flash Attention than Transformer-based TTI models that resemble the Decode phase. Since optimizations designed for LLMs do not map directly onto TTI/TTV models, we must conduct a thorough characterization of these workloads to gain insights for new optimization opportunities. In doing so, we define sequence length in the context of TTI/TTV models and observe sequence length can vary up to 4x in Diffusion model inference. We additionally observe temporal aspects of TTV workloads pose unique system bottlenecks, with Temporal Attention accounting for over 60% of total Attention time. Overall, our in-depth system performance characterization is a critical first step towards designing efficient and deployable systems for emerging TTI/TTV workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02985v1">Can Large Language Models Make the Grade? An Empirical Study Evaluating LLMs Ability to Mark Short Answer Questions in K-12 Education</a></div>
    <div class="paper-meta">
      📅 2024-05-05
    </div>
    <details class="paper-abstract">
      This paper presents reports on a series of experiments with a novel dataset evaluating how well Large Language Models (LLMs) can mark (i.e. grade) open text responses to short answer questions, Specifically, we explore how well different combinations of GPT version and prompt engineering strategies performed at marking real student answers to short answer across different domain areas (Science and History) and grade-levels (spanning ages 5-16) using a new, never-used-before dataset from Carousel, a quizzing platform. We found that GPT-4, with basic few-shot prompting performed well (Kappa, 0.70) and, importantly, very close to human-level performance (0.75). This research builds on prior findings that GPT-4 could reliably score short answer reading comprehension questions at a performance-level very close to that of expert human raters. The proximity to human-level performance, across a variety of subjects and grade levels suggests that LLMs could be a valuable tool for supporting low-stakes formative assessment tasks in K-12 education and has important implications for real-world education delivery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18373v2">Can LLMs Grade Short-Answer Reading Comprehension Questions : An Empirical Study with a Novel Dataset</a></div>
    <div class="paper-meta">
      📅 2024-05-05
    </div>
    <details class="paper-abstract">
      Open-ended questions, which require students to produce multi-word, nontrivial responses, are a popular tool for formative assessment as they provide more specific insights into what students do and don't know. However, grading open-ended questions can be time-consuming leading teachers to resort to simpler question formats or conduct fewer formative assessments. While there has been a longstanding interest in automating of short-answer grading (ASAG), but previous approaches have been technically complex, limiting their use in formative assessment contexts. The newest generation of Large Language Models (LLMs) potentially makes grading short answer questions more feasible. This paper investigates the potential for the newest version of LLMs to be used in ASAG, specifically in the grading of short answer questions for formative assessments, in two ways. First, it introduces a novel dataset of short answer reading comprehension questions, drawn from a set of reading assessments conducted with over 150 students in Ghana. This dataset allows for the evaluation of LLMs in a new context, as they are predominantly designed and trained on data from high-income North American countries. Second, the paper empirically evaluates how well various configurations of generative LLMs grade student short answer responses compared to expert human raters. The findings show that GPT-4, with minimal prompt engineering, performed extremely well on grading the novel dataset (QWK 0.92, F1 0.89), reaching near parity with expert human raters. To our knowledge this work is the first to empirically evaluate the performance of generative LLMs on short answer reading comprehension questions using real student data, with low technical hurdles to attaining this performance. These findings suggest that generative LLMs could be used to grade formative literacy assessment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02858v1">Language Evolution for Evading Social Media Regulation via LLM-based Multi-agent Simulation</a></div>
    <div class="paper-meta">
      📅 2024-05-05
      | 💬 Accepted by IEEE WCCI 2024
    </div>
    <details class="paper-abstract">
      Social media platforms such as Twitter, Reddit, and Sina Weibo play a crucial role in global communication but often encounter strict regulations in geopolitically sensitive regions. This situation has prompted users to ingeniously modify their way of communicating, frequently resorting to coded language in these regulated social media environments. This shift in communication is not merely a strategy to counteract regulation, but a vivid manifestation of language evolution, demonstrating how language naturally evolves under societal and technological pressures. Studying the evolution of language in regulated social media contexts is of significant importance for ensuring freedom of speech, optimizing content moderation, and advancing linguistic research. This paper proposes a multi-agent simulation framework using Large Language Models (LLMs) to explore the evolution of user language in regulated social media environments. The framework employs LLM-driven agents: supervisory agent who enforce dialogue supervision and participant agents who evolve their language strategies while engaging in conversation, simulating the evolution of communication styles under strict regulations aimed at evading social media regulation. The study evaluates the framework's effectiveness through a range of scenarios from abstract scenarios to real-world situations. Key findings indicate that LLMs are capable of simulating nuanced language dynamics and interactions in constrained settings, showing improvement in both evading supervision and information accuracy as evolution progresses. Furthermore, it was found that LLM agents adopt different strategies for different scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.11708v3">Mastering Text-to-Image Diffusion: Recaptioning, Planning, and Generating with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-05
      | 💬 ICML 2024. Project: https://github.com/YangLing0818/RPG-DiffusionMaster
    </div>
    <details class="paper-abstract">
      Diffusion models have exhibit exceptional performance in text-to-image generation and editing. However, existing methods often face challenges when handling complex text prompts that involve multiple objects with multiple attributes and relationships. In this paper, we propose a brand new training-free text-to-image generation/editing framework, namely Recaption, Plan and Generate (RPG), harnessing the powerful chain-of-thought reasoning ability of multimodal LLMs to enhance the compositionality of text-to-image diffusion models. Our approach employs the MLLM as a global planner to decompose the process of generating complex images into multiple simpler generation tasks within subregions. We propose complementary regional diffusion to enable region-wise compositional generation. Furthermore, we integrate text-guided image generation and editing within the proposed RPG in a closed-loop fashion, thereby enhancing generalization ability. Extensive experiments demonstrate our RPG outperforms state-of-the-art text-to-image diffusion models, including DALL-E 3 and SDXL, particularly in multi-category object composition and text-image semantic alignment. Notably, our RPG framework exhibits wide compatibility with various MLLM architectures (e.g., MiniGPT-4) and diffusion backbones (e.g., ControlNet). Our code is available at: https://github.com/YangLing0818/RPG-DiffusionMaster
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02778v1">Improve Temporal Awareness of LLMs for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-05-05
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive zero-shot abilities in solving a wide range of general-purpose tasks. However, it is empirically found that LLMs fall short in recognizing and utilizing temporal information, rendering poor performance in tasks that require an understanding of sequential data, such as sequential recommendation. In this paper, we aim to improve temporal awareness of LLMs by designing a principled prompting framework inspired by human cognitive processes. Specifically, we propose three prompting strategies to exploit temporal information within historical interactions for LLM-based sequential recommendation. Besides, we emulate divergent thinking by aggregating LLM ranking results derived from these strategies. Evaluations on MovieLens-1M and Amazon Review datasets indicate that our proposed method significantly enhances the zero-shot capabilities of LLMs in sequential recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02743v1">Beyond Performance: Quantifying and Mitigating Label Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-04
      | 💬 NAACL 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable adaptability to diverse tasks, by leveraging context prompts containing instructions, or minimal input-output examples. However, recent work revealed they also exhibit label bias -- an undesirable preference toward predicting certain answers over others. Still, detecting and measuring this bias reliably and at scale has remained relatively unexplored. In this study, we evaluate different approaches to quantifying label bias in a model's predictions, conducting a comprehensive investigation across 279 classification tasks and ten LLMs. Our investigation reveals substantial label bias in models both before and after debiasing attempts, as well as highlights the importance of outcomes-based evaluation metrics, which were not previously used in this regard. We further propose a novel label bias calibration method tailored for few-shot prompting, which outperforms recent calibration approaches for both improving performance and mitigating label bias. Our results emphasize that label bias in the predictions of LLMs remains a barrier to their reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.17444v3">LLM-grounded Video Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2024-05-04
      | 💬 ICLR 2024. Project Page: https://llm-grounded-video-diffusion.github.io/
    </div>
    <details class="paper-abstract">
      Text-conditioned diffusion models have emerged as a promising tool for neural video generation. However, current models still struggle with intricate spatiotemporal prompts and often generate restricted or incorrect motion. To address these limitations, we introduce LLM-grounded Video Diffusion (LVD). Instead of directly generating videos from the text inputs, LVD first leverages a large language model (LLM) to generate dynamic scene layouts based on the text inputs and subsequently uses the generated layouts to guide a diffusion model for video generation. We show that LLMs are able to understand complex spatiotemporal dynamics from text alone and generate layouts that align closely with both the prompts and the object motion patterns typically observed in the real world. We then propose to guide video diffusion models with these layouts by adjusting the attention maps. Our approach is training-free and can be integrated into any video diffusion model that admits classifier guidance. Our results demonstrate that LVD significantly outperforms its base video diffusion model and several strong baseline methods in faithfully generating videos with the desired attributes and motion patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01106v2">Distilling Text Style Transfer With Self-Explanation From LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-04
      | 💬 Accepted by NAACL Student Research Workshop 2024
    </div>
    <details class="paper-abstract">
      Text Style Transfer (TST) seeks to alter the style of text while retaining its core content. Given the constraints of limited parallel datasets for TST, we propose CoTeX, a framework that leverages large language models (LLMs) alongside chain-of-thought (CoT) prompting to facilitate TST. CoTeX distills the complex rewriting and reasoning capabilities of LLMs into more streamlined models capable of working with both non-parallel and parallel data. Through experimentation across four TST datasets, CoTeX is shown to surpass traditional supervised fine-tuning and knowledge distillation methods, particularly in low-resource settings. We conduct a comprehensive evaluation, comparing CoTeX against current unsupervised, supervised, in-context learning (ICL) techniques, and instruction-tuned LLMs. Furthermore, CoTeX distinguishes itself by offering transparent explanations for its style transfer process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01103v2">LLM Security Guard for Code</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 SECUTE, EASE 2024
    </div>
    <details class="paper-abstract">
      Many developers rely on Large Language Models (LLMs) to facilitate software development. Nevertheless, these models have exhibited limited capabilities in the security domain. We introduce LLMSecGuard, a framework to offer enhanced code security through the synergy between static code analyzers and LLMs. LLMSecGuard is open source and aims to equip developers with code solutions that are more secure than the code initially generated by LLMs. This framework also has a benchmarking feature, aimed at providing insights into the evolving security attributes of these models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04346v3">Koala: Key frame-conditioned long video-LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Accepted at CVPR 2024 as a poster highlight
    </div>
    <details class="paper-abstract">
      Long video question answering is a challenging task that involves recognizing short-term activities and reasoning about their fine-grained relationships. State-of-the-art video Large Language Models (vLLMs) hold promise as a viable solution due to their demonstrated emergent capabilities on new tasks. However, despite being trained on millions of short seconds-long videos, vLLMs are unable to understand minutes-long videos and accurately answer questions about them. To address this limitation, we propose a lightweight and self-supervised approach, Key frame-conditioned long video-LLM (Koala), that introduces learnable spatiotemporal queries to adapt pretrained vLLMs for generalizing to longer videos. Our approach introduces two new tokenizers that condition on visual tokens computed from sparse video key frames for understanding short and long video moments. We train our proposed approach on HowTo100M and demonstrate its effectiveness on zero-shot long video understanding benchmarks, where it outperforms state-of-the-art large models by 3 - 6% in absolute accuracy across all tasks. Surprisingly, we also empirically show that our approach not only helps a pretrained vLLM to understand long videos but also improves its accuracy on short-term action recognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09128v3">ChainForge: A Visual Toolkit for Prompt Engineering and LLM Hypothesis Testing</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 18 pages, 7 figures, published at CHI 2024
    </div>
    <details class="paper-abstract">
      Evaluating outputs of large language models (LLMs) is challenging, requiring making -- and making sense of -- many responses. Yet tools that go beyond basic prompting tend to require knowledge of programming APIs, focus on narrow domains, or are closed-source. We present ChainForge, an open-source visual toolkit for prompt engineering and on-demand hypothesis testing of text generation LLMs. ChainForge provides a graphical interface for comparison of responses across models and prompt variations. Our system was designed to support three tasks: model selection, prompt template design, and hypothesis testing (e.g., auditing). We released ChainForge early in its development and iterated on its design with academics and online users. Through in-lab and interview studies, we find that a range of people could use ChainForge to investigate hypotheses that matter to them, including in real-world settings. We identify three modes of prompt engineering and LLM hypothesis testing: opportunistic exploration, limited evaluation, and iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00462v2">LUCID: LLM-Generated Utterances for Complex and Interesting Dialogues</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Accepted at NAACL SRW 2024
    </div>
    <details class="paper-abstract">
      Spurred by recent advances in Large Language Models (LLMs), virtual assistants are poised to take a leap forward in terms of their dialogue capabilities. Yet a major bottleneck to achieving genuinely transformative task-oriented dialogue capabilities remains the scarcity of high quality data. Existing datasets, while impressive in scale, have limited domain coverage and contain few genuinely challenging conversational phenomena; those which are present are typically unlabelled, making it difficult to assess the strengths and weaknesses of models without time-consuming and costly human evaluation. Moreover, creating high quality dialogue data has until now required considerable human input, limiting both the scale of these datasets and the ability to rapidly bootstrap data for a new target domain. We aim to overcome these issues with LUCID, a modularised and highly automated LLM-driven data generation system that produces realistic, diverse and challenging dialogues. We use LUCID to generate a seed dataset of 4,277 conversations across 100 intents to demonstrate its capabilities, with a human review finding consistently high quality labels in the generated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.02024v1">Analyzing Narrative Processing in Large Language Models (LLMs): Using GPT4 to test BERT</a></div>
    <div class="paper-meta">
      📅 2024-05-03
    </div>
    <details class="paper-abstract">
      The ability to transmit and receive complex information via language is unique to humans and is the basis of traditions, culture and versatile social interactions. Through the disruptive introduction of transformer based large language models (LLMs) humans are not the only entity to "understand" and produce language any more. In the present study, we have performed the first steps to use LLMs as a model to understand fundamental mechanisms of language processing in neural networks, in order to make predictions and generate hypotheses on how the human brain does language processing. Thus, we have used ChatGPT to generate seven different stylistic variations of ten different narratives (Aesop's fables). We used these stories as input for the open source LLM BERT and have analyzed the activation patterns of the hidden units of BERT using multi-dimensional scaling and cluster analysis. We found that the activation vectors of the hidden units cluster according to stylistic variations in earlier layers of BERT (1) than narrative content (4-5). Despite the fact that BERT consists of 12 identical building blocks that are stacked and trained on large text corpora, the different layers perform different tasks. This is a very useful model of the human brain, where self-similar structures, i.e. different areas of the cerebral cortex, can have different functions and are therefore well suited to processing language in a very efficient way. The proposed approach has the potential to open the black box of LLMs on the one hand, and might be a further step to unravel the neural processes underlying human language processing and cognition in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01926v1">Auto-Encoding Morph-Tokens for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Accepted by ICML 2024
    </div>
    <details class="paper-abstract">
      For multimodal LLMs, the synergy of visual comprehension (textual output) and generation (visual output) presents an ongoing challenge. This is due to a conflicting objective: for comprehension, an MLLM needs to abstract the visuals; for generation, it needs to preserve the visuals as much as possible. Thus, the objective is a dilemma for visual-tokens. To resolve the conflict, we propose encoding images into morph-tokens to serve a dual purpose: for comprehension, they act as visual prompts instructing MLLM to generate texts; for generation, they take on a different, non-conflicting role as complete visual-tokens for image reconstruction, where the missing visual cues are recovered by the MLLM. Extensive experiments show that morph-tokens can achieve a new SOTA for multimodal comprehension and generation simultaneously. Our project is available at https://github.com/DCDmllm/MorphTokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01864v2">(A)I Am Not a Lawyer, But...: Engaging Legal Experts towards Responsible LLM Policies for Legal Advice</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly capable of providing users with advice in a wide range of professional domains, including legal advice. However, relying on LLMs for legal queries raises concerns due to the significant expertise required and the potential real-world consequences of the advice. To explore \textit{when} and \textit{why} LLMs should or should not provide advice to users, we conducted workshops with 20 legal experts using methods inspired by case-based reasoning. The provided realistic queries ("cases") allowed experts to examine granular, situation-specific concerns and overarching technical and legal constraints, producing a concrete set of contextual considerations for LLM developers. By synthesizing the factors that impacted LLM response appropriateness, we present a 4-dimension framework: (1) User attributes and behaviors, (2) Nature of queries, (3) AI capabilities, and (4) Social impacts. We share experts' recommendations for LLM response strategies, which center around helping users identify `right questions to ask' and relevant information rather than providing definitive legal judgments. Our findings reveal novel legal considerations, such as unauthorized practice of law, confidentiality, and liability for inaccurate advice, that have been overlooked in the literature. The case-based deliberation method enabled us to elicit fine-grained, practice-informed insights that surpass those from de-contextualized surveys or speculative principles. These findings underscore the applicability of our method for translating domain-specific professional knowledge and practices into policies that can guide LLM behavior in a more responsible direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01886v1">Aloe: A Family of Fine-tuned Open Healthcare LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Five appendix
    </div>
    <details class="paper-abstract">
      As the capabilities of Large Language Models (LLMs) in healthcare and medicine continue to advance, there is a growing need for competitive open-source models that can safeguard public interest. With the increasing availability of highly competitive open base models, the impact of continued pre-training is increasingly uncertain. In this work, we explore the role of instruct tuning, model merging, alignment, red teaming and advanced inference schemes, as means to improve current open models. To that end, we introduce the Aloe family, a set of open medical LLMs highly competitive within its scale range. Aloe models are trained on the current best base models (Mistral, LLaMA 3), using a new custom dataset which combines public data sources improved with synthetic Chain of Thought (CoT). Aloe models undergo an alignment phase, becoming one of the first few policy-aligned open healthcare LLM using Direct Preference Optimization, setting a new standard for ethical performance in healthcare LLMs. Model evaluation expands to include various bias and toxicity datasets, a dedicated red teaming effort, and a much-needed risk assessment for healthcare LLMs. Finally, to explore the limits of current LLMs in inference, we study several advanced prompt engineering strategies to boost performance across benchmarks, yielding state-of-the-art results for open healthcare 7B LLMs, unprecedented at this scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01883v1">DALLMi: Domain Adaption for LLM-based Multi-label Classifier</a></div>
    <div class="paper-meta">
      📅 2024-05-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as the backbone for classifying text associated with distinct domains and simultaneously several labels (classes). When encountering domain shifts, e.g., classifier of movie reviews from IMDb to Rotten Tomatoes, adapting such an LLM-based multi-label classifier is challenging due to incomplete label sets at the target domain and daunting training overhead. The existing domain adaptation methods address either image multi-label classifiers or text binary classifiers. In this paper, we design DALLMi, Domain Adaptation Large Language Model interpolator, a first-of-its-kind semi-supervised domain adaptation method for text data models based on LLMs, specifically BERT. The core of DALLMi is the novel variation loss and MixUp regularization, which jointly leverage the limited positively labeled and large quantity of unlabeled text and, importantly, their interpolation from the BERT word embeddings. DALLMi also introduces a label-balanced sampling strategy to overcome the imbalance between labeled and unlabeled data. We evaluate DALLMi against the partial-supervised and unsupervised approach on three datasets under different scenarios of label availability for the target domain. Our results show that DALLMi achieves higher mAP than unsupervised and partially-supervised approaches by 19.9% and 52.2%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01868v1">Incorporating External Knowledge and Goal Guidance for LLM-based Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Main paper 8 pages; References and Appendix 9 pages; 7 figures and 14 tables
    </div>
    <details class="paper-abstract">
      This paper aims to efficiently enable large language models (LLMs) to use external knowledge and goal guidance in conversational recommender system (CRS) tasks. Advanced LLMs (e.g., ChatGPT) are limited in domain-specific CRS tasks for 1) generating grounded responses with recommendation-oriented knowledge, or 2) proactively leading the conversations through different dialogue goals. In this work, we first analyze those limitations through a comprehensive evaluation, showing the necessity of external knowledge and goal guidance which contribute significantly to the recommendation accuracy and language quality. In light of this finding, we propose a novel ChatCRS framework to decompose the complex CRS task into several sub-tasks through the implementation of 1) a knowledge retrieval agent using a tool-augmented approach to reason over external Knowledge Bases and 2) a goal-planning agent for dialogue goal prediction. Experimental results on two multi-goal CRS datasets reveal that ChatCRS sets new state-of-the-art benchmarks, improving language quality of informativeness by 17% and proactivity by 27%, and achieving a tenfold enhancement in recommendation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13414v3">Evaluating the Effectiveness of LLMs in Introductory Computer Science Education: A Semester-Long Field Study</a></div>
    <div class="paper-meta">
      📅 2024-05-03
      | 💬 Accepted to Learning @ Scale 2024
    </div>
    <details class="paper-abstract">
      The integration of AI assistants, especially through the development of Large Language Models (LLMs), into computer science education has sparked significant debate. An emerging body of work has looked into using LLMs in education, but few have examined the impacts of LLMs on students in entry-level programming courses, particularly in real-world contexts and over extended periods. To address this research gap, we conducted a semester-long, between-subjects study with 50 students using CodeTutor, an LLM-powered assistant developed by our research team. Our study results show that students who used CodeTutor (the experimental group) achieved statistically significant improvements in their final scores compared to peers who did not use the tool (the control group). Within the experimental group, those without prior experience with LLM-powered tools demonstrated significantly greater performance gain than their counterparts. We also found that students expressed positive feedback regarding CodeTutor's capability, though they also had concerns about CodeTutor's limited role in developing critical thinking skills. Over the semester, students' agreement with CodeTutor's suggestions decreased, with a growing preference for support from traditional human teaching assistants. Our analysis further reveals that the quality of user prompts was significantly correlated with CodeTutor's response effectiveness. Building upon our results, we discuss the implications of our findings for integrating Generative AI literacy into curricula to foster critical thinking skills and turn to examining the temporal dynamics of user engagement with LLM-powered tools. We further discuss the discrepancy between the anticipated functions of tools and students' actual capabilities, which sheds light on the need for tailored strategies to improve educational outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2308.07308v4">LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked</a></div>
    <div class="paper-meta">
      📅 2024-05-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01744v1">ALCM: Autonomous LLM-Augmented Causal Discovery Framework</a></div>
    <div class="paper-meta">
      📅 2024-05-02
    </div>
    <details class="paper-abstract">
      To perform effective causal inference in high-dimensional datasets, initiating the process with causal discovery is imperative, wherein a causal graph is generated based on observational data. However, obtaining a complete and accurate causal graph poses a formidable challenge, recognized as an NP-hard problem. Recently, the advent of Large Language Models (LLMs) has ushered in a new era, indicating their emergent capabilities and widespread applicability in facilitating causal reasoning across diverse domains, such as medicine, finance, and science. The expansive knowledge base of LLMs holds the potential to elevate the field of causal reasoning by offering interpretability, making inferences, generalizability, and uncovering novel causal structures. In this paper, we introduce a new framework, named Autonomous LLM-Augmented Causal Discovery Framework (ALCM), to synergize data-driven causal discovery algorithms and LLMs, automating the generation of a more resilient, accurate, and explicable causal graph. The ALCM consists of three integral components: causal structure learning, causal wrapper, and LLM-driven causal refiner. These components autonomously collaborate within a dynamic environment to address causal discovery questions and deliver plausible causal graphs. We evaluate the ALCM framework by implementing two demonstrations on seven well-known datasets. Experimental results demonstrate that ALCM outperforms existing LLM methods and conventional data-driven causal reasoning mechanisms. This study not only shows the effectiveness of the ALCM but also underscores new research directions in leveraging the causal reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01695v1">Requirements-driven Slicing of Simulink Models Using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 This paper will appear at the 11th International Workshop on Artificial Intelligence and Requirements Engineering (AIRE'24)
    </div>
    <details class="paper-abstract">
      Model slicing is a useful technique for identifying a subset of a larger model that is relevant to fulfilling a given requirement. Notable applications of slicing include reducing inspection effort when checking design adequacy to meet requirements of interest and when conducting change impact analysis. In this paper, we present a method based on large language models (LLMs) for extracting model slices from graphical Simulink models. Our approach converts a Simulink model into a textual representation, uses an LLM to identify the necessary Simulink blocks for satisfying a specific requirement, and constructs a sound model slice that incorporates the blocks identified by the LLM. We explore how different levels of granularity (verbosity) in transforming Simulink models into textual representations, as well as the strategy used to prompt the LLM, impact the accuracy of the generated slices. Our preliminary findings suggest that prompts created by textual representations that retain the syntax and semantics of Simulink blocks while omitting visual rendering information of Simulink models yield the most accurate slices. Furthermore, the chain-of-thought and zero-shot prompting strategies result in the largest number of accurate model slices produced by our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01668v1">WitheredLeaf: Finding Entity-Inconsistency Bugs with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-05-02
    </div>
    <details class="paper-abstract">
      Originating from semantic bugs, Entity-Inconsistency Bugs (EIBs) involve misuse of syntactically valid yet incorrect program entities, such as variable identifiers and function names, which often have security implications. Unlike straightforward syntactic vulnerabilities, EIBs are subtle and can remain undetected for years. Traditional detection methods, such as static analysis and dynamic testing, often fall short due to the versatile and context-dependent nature of EIBs. However, with advancements in Large Language Models (LLMs) like GPT-4, we believe LLM-powered automatic EIB detection becomes increasingly feasible through these models' semantics understanding abilities. This research first undertakes a systematic measurement of LLMs' capabilities in detecting EIBs, revealing that GPT-4, while promising, shows limited recall and precision that hinder its practical application. The primary problem lies in the model's tendency to focus on irrelevant code snippets devoid of EIBs. To address this, we introduce a novel, cascaded EIB detection system named WitheredLeaf, which leverages smaller, code-specific language models to filter out most negative cases and mitigate the problem, thereby significantly enhancing the overall precision and recall. We evaluated WitheredLeaf on 154 Python and C GitHub repositories, each with over 1,000 stars, identifying 123 new flaws, 45% of which can be exploited to disrupt the program's normal operations. Out of 69 submitted fixes, 27 have been successfully merged.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01533v1">OmniDrive: A Holistic LLM-Agent Framework for Autonomous Driving with 3D Perception, Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2024-05-02
    </div>
    <details class="paper-abstract">
      The advances in multimodal large language models (MLLMs) have led to growing interests in LLM-based autonomous driving agents to leverage their strong reasoning capabilities. However, capitalizing on MLLMs' strong reasoning capabilities for improved planning behavior is challenging since planning requires full 3D situational awareness beyond 2D reasoning. To address this challenge, our work proposes a holistic framework for strong alignment between agent models and 3D driving tasks. Our framework starts with a novel 3D MLLM architecture that uses sparse queries to lift and compress visual representations into 3D before feeding them into an LLM. This query-based representation allows us to jointly encode dynamic objects and static map elements (e.g., traffic lanes), providing a condensed world model for perception-action alignment in 3D. We further propose OmniDrive-nuScenes, a new visual question-answering dataset challenging the true 3D situational awareness of a model with comprehensive visual question-answering (VQA) tasks, including scene description, traffic regulation, 3D grounding, counterfactual reasoning, decision making and planning. Extensive studies show the effectiveness of the proposed architecture as well as the importance of the VQA tasks for reasoning and planning in complex 3D scenes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.07308v4">LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked</a></div>
    <div class="paper-meta">
      📅 2024-05-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01310v1">Overcoming LLM Challenges using RAG-Driven Precision in Coffee Leaf Disease Remediation</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This research introduces an innovative AI-driven precision agriculture system, leveraging YOLOv8 for disease identification and Retrieval Augmented Generation (RAG) for context-aware diagnosis. Focused on addressing the challenges of diseases affecting the coffee production sector in Karnataka, The system integrates sophisticated object detection techniques with language models to address the inherent constraints associated with Large Language Models (LLMs). Our methodology not only tackles the issue of hallucinations in LLMs, but also introduces dynamic disease identification and remediation strategies. Real-time monitoring, collaborative dataset expansion, and organizational involvement ensure the system's adaptability in diverse agricultural settings. The effect of the suggested system extends beyond automation, aiming to secure food supplies, protect livelihoods, and promote eco-friendly farming practices. By facilitating precise disease identification, the system contributes to sustainable and environmentally conscious agriculture, reducing reliance on pesticides. Looking to the future, the project envisions continuous development in RAG-integrated object detection systems, emphasizing scalability, reliability, and usability. This research strives to be a beacon for positive change in agriculture, aligning with global efforts toward sustainable and technologically enhanced food production.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01299v1">The Effectiveness of LLMs as Annotators: A Comparative Overview and Empirical Analysis of Direct Representation</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 LREC-COLING NLPerspectives workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful support tools across various natural language tasks and a range of application domains. Recent studies focus on exploring their capabilities for data annotation. This paper provides a comparative overview of twelve studies investigating the potential of LLMs in labelling data. While the models demonstrate promising cost and time-saving benefits, there exist considerable limitations, such as representativeness, bias, sensitivity to prompt variations and English language preference. Leveraging insights from these studies, our empirical analysis further examines the alignment between human and GPT-generated opinion distributions across four subjective datasets. In contrast to the studies examining representation, our methodology directly obtains the opinion distribution from GPT. Our analysis thereby supports the minority of studies that are considering diverse perspectives when evaluating data annotation tasks and highlights the need for further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.03150v2">Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly</a></div>
    <div class="paper-meta">
      📅 2024-05-02
      | 💬 Camera-ready version for DEEM'24. Please cite the official ACM paper via https://doi.org/10.1145/3650203.3663331
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) and foundation models are popular as they offer new opportunities for individuals and businesses to improve natural language processing, interact with data, and retrieve information faster. However, training or fine-tuning LLMs requires a vast amount of data, which can be challenging to access due to legal or technical restrictions and may require private computing resources. Federated Learning (FL) is a solution designed to overcome these challenges and expand data access for deep learning applications. This paper takes a hardware-centric approach to explore how LLMs can be brought to modern edge computing systems. Our study fine-tunes the FLAN-T5 model family, ranging from 80M to 3B parameters, using FL for a text summarization task. We provide a micro-level hardware benchmark, compare the model FLOP utilization to a state-of-the-art data center GPU, and study the network utilization in realistic conditions. Our contribution is twofold: First, we evaluate the current capabilities of edge computing systems and their potential for LLM FL workloads. Second, by comparing these systems with a data-center GPU, we demonstrate the potential for improvement and the next steps toward achieving greater computational efficiency at the edge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2308.07308v4">LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked</a></div>
    <div class="paper-meta">
      📅 2024-05-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2. The code is publicly available at https://github.com/poloclub/llm-self-defense
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16363v6">LLM Inference Unveiled: Survey and Roofline Model Insights</a></div>
    <div class="paper-meta">
      📅 2024-05-01
    </div>
    <details class="paper-abstract">
      The field of efficient Large Language Model (LLM) inference is rapidly evolving, presenting a unique blend of opportunities and challenges. Although the field has expanded and is vibrant, there hasn't been a concise framework that analyzes the various methods of LLM Inference to provide a clear understanding of this domain. Our survey stands out from traditional literature reviews by not only summarizing the current state of research but also by introducing a framework based on roofline model for systematic analysis of LLM inference techniques. This framework identifies the bottlenecks when deploying LLMs on hardware devices and provides a clear understanding of practical problems, such as why LLMs are memory-bound, how much memory and computation they need, and how to choose the right hardware. We systematically collate the latest advancements in efficient LLM inference, covering crucial areas such as model compression (e.g., Knowledge Distillation and Quantization), algorithm improvements (e.g., Early Exit and Mixture-of-Expert), and both hardware and system-level enhancements. Our survey stands out by analyzing these methods with roofline model, helping us understand their impact on memory access and computation. This distinctive approach not only showcases the current research landscape but also delivers valuable insights for practical implementation, positioning our work as an indispensable resource for researchers new to the field as well as for those seeking to deepen their understanding of efficient LLM deployment. The analyze tool, LLM-Viewer, is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.18796v2">Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models</a></div>
    <div class="paper-meta">
      📅 2024-05-01
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have become more advanced, they have outpaced our abilities to accurately evaluate their quality. Not only is finding data to adequately probe particular model properties difficult, but evaluating the correctness of a model's freeform generation alone is a challenge. To address this, many evaluations now rely on using LLMs themselves as judges to score the quality of outputs from other LLMs. Evaluations most commonly use a single large model like GPT4. While this method has grown in popularity, it is costly, has been shown to introduce intramodel bias, and in this work, we find that very large models are often unnecessary. We propose instead to evaluate models using a Panel of LLm evaluators (PoLL). Across three distinct judge settings and spanning six different datasets, we find that using a PoLL composed of a larger number of smaller models outperforms a single large judge, exhibits less intra-model bias due to its composition of disjoint model families, and does so while being over seven times less expensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00467v1">Harnessing the Power of Multiple Minds: Lessons Learned from LLM Routing</a></div>
    <div class="paper-meta">
      📅 2024-05-01
      | 💬 Accepted to Workshop on Insights from Negative Results in NLP 2024 (co-located with NAACL 2024)
    </div>
    <details class="paper-abstract">
      With the rapid development of LLMs, it is natural to ask how to harness their capabilities efficiently. In this paper, we explore whether it is feasible to direct each input query to a single most suitable LLM. To this end, we propose LLM routing for challenging reasoning tasks. Our extensive experiments suggest that such routing shows promise but is not feasible in all scenarios, so more robust approaches should be investigated to fill this gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09801v2">ProCoT: Stimulating Critical Thinking and Writing of Students through Engagement with Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-05-01
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      We introduce a novel writing method called Probing Chain-of-Thought (ProCoT), which potentially prevents students from cheating using a Large Language Model (LLM), such as ChatGPT, while enhancing their active learning. LLMs have disrupted education and many other fields. For fear of students cheating, many have resorted to banning their use. These LLMs are also known for hallucinations. We conduct studies with ProCoT in two different courses with 65 students. The students in each course were asked to prompt an LLM of their choice with one question from a set of four and required to affirm or refute statements in the LLM output by using peer-reviewed references. The results show two things: (1) ProCoT stimulates creative/critical thinking and writing of students through engagement with LLMs when we compare the LLM-only output to ProCoT output and (2) ProCoT can prevent cheating because of clear limitations in existing LLMs, particularly ChatGPT, when we compare students' ProCoT output to LLM ProCoT output. We also discover that most students prefer to give answers in fewer words than LLMs, which are typically verbose. The average word counts for students in the first course, ChatGPT (v3.5), and Phind (v8) are 208, 391 and 383, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00321v1">DFKI-NLP at SemEval-2024 Task 2: Towards Robust LLMs Using Data Perturbations and MinMax Training</a></div>
    <div class="paper-meta">
      📅 2024-05-01
    </div>
    <details class="paper-abstract">
      The NLI4CT task at SemEval-2024 emphasizes the development of robust models for Natural Language Inference on Clinical Trial Reports (CTRs) using large language models (LLMs). This edition introduces interventions specifically targeting the numerical, vocabulary, and semantic aspects of CTRs. Our proposed system harnesses the capabilities of the state-of-the-art Mistral model, complemented by an auxiliary model, to focus on the intricate input space of the NLI4CT dataset. Through the incorporation of numerical and acronym-based perturbations to the data, we train a robust system capable of handling both semantic-altering and numerical contradiction interventions. Our analysis on the dataset sheds light on the challenging sections of the CTRs for reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19097v2">Exploring the Capability of LLMs in Performing Low-Level Visual Analytic Tasks on SVG Data Visualizations</a></div>
    <div class="paper-meta">
      📅 2024-05-01
    </div>
    <details class="paper-abstract">
      Data visualizations help extract insights from datasets, but reaching these insights requires decomposing high level goals into low-level analytic tasks that can be complex due to varying degrees of data literacy and visualization experience. Recent advancements in large language models (LLMs) have shown promise for lowering barriers for users to achieve tasks such as writing code and may likewise facilitate visualization insight. Scalable Vector Graphics (SVG), a text-based image format common in data visualizations, matches well with the text sequence processing of transformer-based LLMs. In this paper, we explore the capability of LLMs to perform 10 low-level visual analytic tasks defined by Amar, Eagan, and Stasko directly on SVG-based visualizations. Using zero-shot prompts, we instruct the models to provide responses or modify the SVG code based on given visualizations. Our findings demonstrate that LLMs can effectively modify existing SVG visualizations for some tasks like Cluster but perform poorly on tasks requiring mathematical operations like Compute Derived Value. We also discovered that LLM performance can vary based on factors such as the number of data points, the presence of value labels, and the chart type. Our findings contribute to gauging the general capabilities of LLMs and highlight the need for further exploration and development to fully harness their potential in supporting visual analytic tasks.
    </details>
</div>
