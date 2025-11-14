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
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12214v1">Zero Token-Driven Deep Thinking in LLMs: Unlocking the Full Potential of Existing Parameters via Cyclic Refinement</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Resource limitations often constrain the parameter counts of Large Language Models (LLMs), hindering their performance. While existing methods employ parameter sharing to reuse the same parameter set under fixed budgets, such approaches typically force each layer to assume multiple roles with a predetermined number of iterations, restricting efficiency and adaptability. In this work, we propose the Zero Token Transformer (ZTT), which features a head-tail decoupled parameter cycling method. We disentangle the first (head) and last (tail) layers from parameter cycling and iteratively refine only the intermediate layers. Furthermore, we introduce a Zero-Token Mechanism, an internal architectural component rather than an input token, to guide layer-specific computation. At each cycle, the model retrieves a zero token (with trainable key values) from a Zero-Token Pool, integrating it alongside regular tokens in the attention mechanism. The corresponding attention scores not only reflect each layer's computational importance but also enable dynamic early exits without sacrificing overall model accuracy. Our approach achieves superior performance under tight parameter budgets, effectively reduces computational overhead via early exits, and can be readily applied to fine-tune existing pre-trained models for enhanced efficiency and adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12134v1">SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning enables Large Language Models (LLMs) to solve complex reasoning tasks by generating intermediate reasoning steps. However, most existing approaches focus on hard token decoding, which constrains reasoning within the discrete vocabulary space and may not always be optimal. While recent efforts explore continuous-space reasoning, they often suffer from catastrophic forgetting, limiting their applicability to state-of-the-art LLMs that already perform well in zero-shot settings with a proper instruction. To address this challenge, we propose a novel approach for continuous-space reasoning that does not require modifying the underlying LLM. Specifically, we employ a lightweight assistant model to generate instance-specific soft thought tokens speculatively as the initial chain of thoughts, which are then mapped into the LLM's representation space via a projection module. Experimental results on five reasoning benchmarks demonstrate that our method enhances LLM reasoning performance through supervised, parameter-efficient fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09606v2">Human-LLM Coevolution: Evidence from Academic Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      With a statistical analysis of arXiv paper abstracts, we report a marked drop in the frequency of several words previously identified as overused by ChatGPT, such as "delve", starting soon after they were pointed out in early 2024. The frequency of certain other words favored by ChatGPT, such as "significant", has instead kept increasing. These phenomena suggest that some authors of academic papers have adapted their use of large language models (LLMs), for example, by selecting outputs or applying modifications to the LLM-generated content. Such coevolution and cooperation of humans and LLMs thus introduce additional challenges to the detection of machine-generated text in real-world scenarios. Estimating the impact of LLMs on academic writing by examining word frequency remains feasible, and more attention should be paid to words that were already frequently employed, including those that have decreased in frequency due to LLMs' disfavor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12120v1">LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v1">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code is available at https://github.com/WujiangXu/AgenticMemory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03823v2">Both Text and Images Leaked! A Systematic Analysis of Multimodal LLM Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Code Available: https://github.com/MLLM-Data-Contamination/MM-Detect
    </div>
    <details class="paper-abstract">
      The rapid progression of multimodal large language models (MLLMs) has demonstrated superior performance on various multimodal benchmarks. However, the issue of data contamination during training creates challenges in performance evaluation and comparison. While numerous methods exist for detecting models' contamination in large language models (LLMs), they are less effective for MLLMs due to their various modalities and multiple training phases. In this study, we introduce a multimodal data contamination detection framework, MM-Detect, designed for MLLMs. Our experimental results indicate that MM-Detect is quite effective and sensitive in identifying varying degrees of contamination, and can highlight significant performance improvements due to the leakage of multimodal benchmark training sets. Furthermore, we explore whether the contamination originates from the base LLMs used by MLLMs or the multimodal training phase, providing new insights into the stages at which contamination may be introduced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16491v2">BIG5-CHAT: Shaping LLM Personalities Through Training on Human-Grounded Data</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of embedding realistic human personality traits into LLMs. Previous approaches have primarily focused on prompt-based methods that describe the behavior associated with the desired personality traits, suffering from realism and validity issues. To address these limitations, we introduce BIG5-CHAT, a large-scale dataset containing 100,000 dialogues designed to ground models in how humans express their personality in language. Leveraging this dataset, we explore Supervised Fine-Tuning and Direct Preference Optimization as training-based methods to align LLMs more naturally with human personality patterns. Our methods outperform prompting on personality assessments such as BFI and IPIP-NEO, with trait correlations more closely matching human data. Furthermore, our experiments reveal that models trained to exhibit higher conscientiousness, higher agreeableness, lower extraversion, and lower neuroticism display better performance on reasoning tasks, aligning with psychological findings on how these traits impact human cognitive performance. To our knowledge, this work is the first comprehensive study to demonstrate how training-based methods can shape LLM personalities through learning from real human behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12073v1">Can LLMs Simulate Social Media Engagement? A Study on Action-Guided Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Social media enables dynamic user engagement with trending topics, and recent research has explored the potential of large language models (LLMs) for response generation. While some studies investigate LLMs as agents for simulating user behavior on social media, their focus remains on practical viability and scalability rather than a deeper understanding of how well LLM aligns with human behavior. This paper analyzes LLMs' ability to simulate social media engagement through action guided response generation, where a model first predicts a user's most likely engagement action-retweet, quote, or rewrite-towards a trending post before generating a personalized response conditioned on the predicted action. We benchmark GPT-4o-mini, O1-mini, and DeepSeek-R1 in social media engagement simulation regarding a major societal event discussed on X. Our findings reveal that zero-shot LLMs underperform BERT in action prediction, while few-shot prompting initially degrades the prediction accuracy of LLMs with limited examples. However, in response generation, few-shot LLMs achieve stronger semantic alignment with ground truth posts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v1">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12065v1">Formalizing Complex Mathematical Statements with LLMs: A Study on Mathematical Definitions</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge the gap between informal mathematics and formal languages through autoformalization. However, it is still unclear how well LLMs generalize to sophisticated and naturally occurring mathematical statements. To address this gap, we investigate the task of autoformalizing real-world mathematical definitions -- a critical component of mathematical discourse. Specifically, we introduce two novel resources for autoformalisation, collecting definitions from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically evaluate a range of LLMs, analyzing their ability to formalize definitions into Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs' performance including refinement through external feedback from Proof Assistants, and formal definition grounding, where we guide LLMs through relevant contextual elements from formal mathematical libraries. Our findings reveal that definitions present a greater challenge compared to existing benchmarks, such as miniF2F. In particular, we found that LLMs still struggle with self-correction, and aligning with relevant mathematical libraries. At the same time, structured refinement methods and definition grounding strategies yield notable improvements of up to 16% on self-correction capabilities and 43% on the reduction of undefined errors, highlighting promising directions for enhancing LLM-based autoformalization in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12055v1">Designing Role Vectors to Improve LLM Inference Behaviour</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Submitted to ARR 2025 February cycle
    </div>
    <details class="paper-abstract">
      The influence of personas on Large Language Models (LLMs) has been widely studied, yet their direct impact on performance remains uncertain. This work explores a novel approach to guiding LLM behaviour through role vectors, an alternative to persona-based prompting. We construct 29 role vectors derived from model activations and evaluate their impact on benchmark performance across multiple domains. Our analysis investigates whether these vectors can effectively steer models toward domain-specific expertise. We measure two key interventions: (i) activation addition, which reinforces role-specific directions, and (ii) directional ablation, which removes them. Results on well-established benchmarks indicate that role vectors do, in fact, influence model behaviour, improving task performance in relevant domains while marginally affecting unrelated tasks. This, in turn, suggests that manipulating internal model representations has a greater impact on outcomes than persona-based prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12029v1">KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in various complex tasks, yet they still suffer from hallucinations. Introducing external knowledge, such as knowledge graph, can enhance the LLMs' ability to provide factual answers. LLMs have the ability to interactively explore knowledge graphs. However, most approaches have been affected by insufficient internal knowledge excavation in LLMs, limited generation of trustworthy knowledge reasoning paths, and a vague integration between internal and external knowledge. Therefore, we propose KnowPath, a knowledge-enhanced large model framework driven by the collaboration of internal and external knowledge. It relies on the internal knowledge of the LLM to guide the exploration of interpretable directed subgraphs in external knowledge graphs, better integrating the two knowledge sources for more accurate reasoning. Extensive experiments on multiple real-world datasets confirm the superiority of KnowPath.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12022v1">Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.20856v2">Strada-LLM: Graph LLM for traffic prediction</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 The reviewers decided to reject it. After getting the reviews, we wanted to study more.
    </div>
    <details class="paper-abstract">
      Traffic prediction is a vital component of intelligent transportation systems. By reasoning about traffic patterns in both the spatial and temporal dimensions, accurate and interpretable predictions can be provided. A considerable challenge in traffic prediction lies in handling the diverse data distributions caused by vastly different traffic conditions occurring at different locations. LLMs have been a dominant solution due to their remarkable capacity to adapt to new datasets with very few labeled data samples, i.e., few-shot adaptability. However, existing forecasting techniques mainly focus on extracting local graph information and forming a text-like prompt, leaving LLM- based traffic prediction an open problem. This work presents a probabilistic LLM for traffic forecasting with three highlights. We propose a graph-aware LLM for traffic prediction that considers proximal traffic information. Specifically, by considering the traffic of neighboring nodes as covariates, our model outperforms the corresponding time-series LLM. Furthermore, we adopt a lightweight approach for efficient domain adaptation when facing new data distributions in few-shot fashion. The comparative experiment demonstrates the proposed method outperforms the state-of-the-art LLM-based methods and the traditional GNN- based supervised approaches. Furthermore, Strada-LLM can be easily adapted to different LLM backbones without a noticeable performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11863v1">FedEAT: A Robustness Optimization Framework for Federated LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11306v1">Smoothing Out Hallucinations: Mitigating LLM Hallucination with Smoothed Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from hallucination, generating factually incorrect or ungrounded content, which limits their reliability in high-stakes applications. A key factor contributing to hallucination is the use of hard labels during training, which enforce deterministic supervision, encourage overconfidence, and disregard the uncertainty inherent in natural language. To address this, we propose mitigating hallucination through knowledge distillation (KD), where a teacher model provides smoothed soft labels to a student model, reducing overconfidence and improving factual grounding. We apply KD during supervised finetuning on instructional data, evaluating its effectiveness across LLMs from different families. Experimental results on summarization benchmarks demonstrate that KD reduces hallucination compared to standard finetuning while preserving performance on general NLP tasks. These findings highlight KD as a promising approach for mitigating hallucination in LLMs and improving model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11304v1">Leveraging Multimodal-LLMs Assisted by Instance Segmentation for Intelligent Traffic Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 6 pages, 7 figures, submitted to 30th IEEE International Symposium on Computers and Communications (ISCC) 2025
    </div>
    <details class="paper-abstract">
      A robust and efficient traffic monitoring system is essential for smart cities and Intelligent Transportation Systems (ITS), using sensors and cameras to track vehicle movements, optimize traffic flow, reduce congestion, enhance road safety, and enable real-time adaptive traffic control. Traffic monitoring models must comprehensively understand dynamic urban conditions and provide an intuitive user interface for effective management. This research leverages the LLaVA visual grounding multimodal large language model (LLM) for traffic monitoring tasks on the real-time Quanser Interactive Lab simulation platform, covering scenarios like intersections, congestion, and collisions. Cameras placed at multiple urban locations collect real-time images from the simulation, which are fed into the LLaVA model with queries for analysis. An instance segmentation model integrated into the cameras highlights key elements such as vehicles and pedestrians, enhancing training and throughput. The system achieves 84.3% accuracy in recognizing vehicle locations and 76.4% in determining steering direction, outperforming traditional models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17983v3">Is The Watermarking Of LLM-Generated Code Robust?</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      We present the first in depth study on the robustness of existing watermarking techniques applied to code generated by large language models (LLMs). As LLMs increasingly contribute to software development, watermarking has emerged as a potential solution for detecting AI generated code and mitigating misuse, such as plagiarism or the automated generation of malicious programs. While previous research has demonstrated the resilience of watermarking in the text setting, our work reveals that watermarking techniques are significantly more fragile in code-based contexts. Specifically, we show that simple semantic-preserving transformations, such as variable renaming and dead code insertion, can effectively erase watermarks without altering the program's functionality. To systematically evaluate watermark robustness, we develop an algorithm that traverses the Abstract Syntax Tree (AST) of a watermarked program and applies a sequence of randomized, semantics-preserving transformations. Our experimental results, conducted on Python code generated by different LLMs, indicate that even minor modifications can drastically reduce watermark detectability, with true positive rates (TPR) dropping below 50% in many cases. Our code is publicly available at https://github.com/uiuc-arc/llm-code-watermark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10863v2">Exploring the Personality Traits of LLMs through Latent Features Steering</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced dialogue systems and role-playing agents through their ability to generate human-like text. While prior studies have shown that LLMs can exhibit distinct and consistent personalities, the mechanisms through which these models encode and express specific personality traits remain poorly understood. To address this, we investigate how various factors, such as cultural norms and environmental stressors, encoded within LLMs, shape their personality traits, guided by the theoretical framework of social determinism. Inspired by related work on LLM interpretability, we propose a training-free approach to modify the model's behavior by extracting and steering latent features corresponding to factors within the model, thereby eliminating the need for retraining. Furthermore, we analyze the implications of these factors for model safety, focusing on their impact through the lens of personality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11275v1">Cuckoo: An IE Free Rider Hatched by Massive Nutrition in LLM's Nest</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Massive high-quality data, both pre-training raw texts and post-training annotations, have been carefully prepared to incubate advanced large language models (LLMs). In contrast, for information extraction (IE), pre-training data, such as BIO-tagged sequences, are hard to scale up. We show that IE models can act as free riders on LLM resources by reframing next-token \emph{prediction} into \emph{extraction} for tokens already present in the context. Specifically, our proposed next tokens extraction (NTE) paradigm learns a versatile IE model, \emph{Cuckoo}, with 102.6M extractive data converted from LLM's pre-training and post-training data. Under the few-shot setting, Cuckoo adapts effectively to traditional and complex instruction-following IE with better performance than existing pre-trained IE models. As a free rider, Cuckoo can naturally evolve with the ongoing advancements in LLM data preparation, benefiting from improvements in LLM training pipelines without additional manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11242v1">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.05345v2">QuaLLM: An LLM-based Framework to Extract Quantitative Insights from Online Forums</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Accepted to NAACL Findings (2025), cite appropriately. Preliminary version presented at CHI LLM as Research Tools Workshop (2024)
    </div>
    <details class="paper-abstract">
      Online discussion forums provide crucial data to understand the concerns of a wide range of real-world communities. However, the typical qualitative and quantitative methodologies used to analyze those data, such as thematic analysis and topic modeling, are infeasible to scale or require significant human effort to translate outputs to human readable forms. This study introduces QuaLLM, a novel LLM-based framework to analyze and extract quantitative insights from text data on online forums. The framework consists of a novel prompting and human evaluation methodology. We applied this framework to analyze over one million comments from two of Reddit's rideshare worker communities, marking the largest study of its type. We uncover significant worker concerns regarding AI and algorithmic platform decisions, responding to regulatory calls about worker insights. In short, our work sets a new precedent for AI-assisted quantitative data analysis to surface concerns from online forums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11228v1">Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 A RAG pipeline that accounts for both diversity and answer quality and that can be used with any LLM backbone to solve complex multi-hop question-answering tasks
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) for domain-specific question-answering (QA) tasks by leveraging external knowledge sources. However, traditional RAG systems primarily focus on relevance-based retrieval and often struggle with redundancy, especially when reasoning requires connecting information from multiple sources. This paper introduces Vendi-RAG, a framework based on an iterative process that jointly optimizes retrieval diversity and answer quality. This joint optimization leads to significantly higher accuracy for multi-hop QA tasks. Vendi-RAG leverages the Vendi Score (VS), a flexible similarity-based diversity metric, to promote semantic diversity in document retrieval. It then uses an LLM judge that evaluates candidate answers, generated after a reasoning step, and outputs a score that the retriever uses to balance relevance and diversity among the retrieved documents during each iteration. Experiments on three challenging datasets -- HotpotQA, MuSiQue, and 2WikiMultiHopQA -- demonstrate Vendi-RAG's effectiveness in multi-hop reasoning tasks. The framework achieves significant accuracy improvements over traditional single-step and multi-step RAG approaches, with accuracy increases reaching up to +4.2% on HotpotQA, +4.1% on 2WikiMultiHopQA, and +1.3% on MuSiQue compared to Adaptive-RAG, the current best baseline. The benefits of Vendi-RAG are even more pronounced as the number of retrieved documents increases. Finally, we evaluated Vendi-RAG across different LLM backbones, including GPT-3.5, GPT-4, and GPT-4o-mini, and observed consistent improvements, demonstrating that the framework's advantages are model-agnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07046v2">SnipGen: A Mining Repository Framework for Evaluating LLMs for Code</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 5 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Language Models (LLMs), such as transformer-based neural networks trained on billions of parameters, have become increasingly prevalent in software engineering (SE). These models, trained on extensive datasets that include code repositories, exhibit remarkable capabilities for SE tasks. However, evaluating their effectiveness poses significant challenges, primarily due to the potential overlap between the datasets used for training and those employed for evaluation. To address this issue, we introduce SnipGen, a comprehensive repository mining framework designed to leverage prompt engineering across various downstream tasks for code generation. SnipGen aims to mitigate data contamination by generating robust testbeds and crafting tailored data points to assist researchers and practitioners in evaluating LLMs for code-related tasks. In our exploratory study, SnipGen mined approximately 227K data points from 338K recent code changes in GitHub commits, focusing on method-level granularity. SnipGen features a collection of prompt templates that can be combined to create a Chain-of-Thought-like sequence of prompts, enabling a nuanced assessment of LLMs' code generation quality. By providing the mining tool, the methodology, and the dataset, SnipGen empowers researchers and practitioners to rigorously evaluate and interpret LLMs' performance in software engineering contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18038v2">POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '25), March 30 - April 3, 2025, Rotterdam, Netherlands
    </div>
    <details class="paper-abstract">
      Each request in LLM inference goes through two phases: compute-bound prefill and memory-bandwidth-bound decode. To improve GPU utilization, recent systems use hybrid batching that combines the prefill and decode phases of different requests into the same batch. This approach optimizes linear operations but remains inefficient for attention computation because existing attention kernels specialize execution independently for the prefill and decode phases. In this paper, we present POD-Attention - the first GPU kernel that efficiently computes attention for hybrid batches. POD-Attention aims to maximize the utilization of both compute and memory bandwidth by carefully allocating the GPU's resources such that prefill and decode operations happen concurrently on the same multiprocessor. POD-Attention speeds up attention computation by up to $59\%$ (mean $28\%$), enabling higher throughput and lower latency LLM inference compared to the use of independently optimized prefill and decode attention kernels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11223v1">Asymmetric Conflict and Synergy in Post-training for LLM-based Multilingual Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has advanced the multilingual machine translation (MMT), yet the Curse of Multilinguality (CoM) remains a major challenge. Existing work in LLM-based MMT typically mitigates this issue via scaling up training and computation budget, which raises a critical question: Is scaling up the training and computation budget truly necessary for high-quality MMT, or can a deeper understanding of CoM provide a more efficient solution? To explore this problem, we analyze the linguistic conflicts and synergy, the underlying mechanism of CoM during post-training phase. We identify an asymmetric phenomenon in linguistic conflicts and synergy: the dominance of conflicts and synergy varies in different translation directions, leading to sub-optimal adaptation in existing post-training methods. We further find that a significant bottleneck in MMT appears to lie in post-training rather than multilingual pre-training, suggesting the need for more effective adaptation strategies. Building on these new insights, we propose a direction-aware training approach, combined with group-wise model merging, to address asymmetry in linguistic conflicts and synergy explicitly. Leveraging this strategy, our method fine-tunes X-ALMA-13B-Pretrain-trained only with multilingual pre-training-achieving comparable performance to XALMA-13B (only SFT) while using only 20B pretraining tokens and 17B parameters-5.5x fewer pretraining-tokens and 1.7x fewer model size-with just 0.85 COMET drop on Flores-200 testsets of 50 languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11221v1">PlanGenLLMs: A Modern Survey of LLM Planning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11211v1">A Survey of LLM-based Agents in Medicine: How far are we from Baymax?</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02305v2">CRMArena: Understanding the Capacity of LLM Agents to Perform Professional CRM Tasks in Realistic Environments</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Customer Relationship Management (CRM) systems are vital for modern enterprises, providing a foundation for managing customer interactions and data. Integrating AI agents into CRM systems can automate routine processes and enhance personalized service. However, deploying and evaluating these agents is challenging due to the lack of realistic benchmarks that reflect the complexity of real-world CRM tasks. To address this issue, we introduce CRMArena, a novel benchmark designed to evaluate AI agents on realistic tasks grounded in professional work environments. Following guidance from CRM experts and industry best practices, we designed CRMArena with nine customer service tasks distributed across three personas: service agent, analyst, and manager. The benchmark includes 16 commonly used industrial objects (e.g., account, order, knowledge article, case) with high interconnectivity, along with latent variables (e.g., complaint habits, policy violations) to simulate realistic data distributions. Experimental results reveal that state-of-the-art LLM agents succeed in less than 40% of the tasks with ReAct prompting, and less than 55% even with function-calling abilities. Our findings highlight the need for enhanced agent capabilities in function-calling and rule-following to be deployed in real-world work environments. CRMArena is an open challenge to the community: systems that can reliably complete tasks showcase direct business value in a popular work environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11196v1">How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at https://github.com/zjunlp/DynamicKnowledgeCircuits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11191v1">Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.88% improvement in the aggregate score, while reasoning distillation leads to a 10% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11187v1">TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 LLMs, Benchmarking, Large Language Models, Bangla
    </div>
    <details class="paper-abstract">
      In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1B and 3B parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately 37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to evaluate LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (https://huggingface.co/collections/hishab/titulm-llama-family-6718d31fc1b83529276f490a).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11183v1">Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Recent advancements in tree search algorithms guided by verifiers have significantly enhanced the reasoning capabilities of large language models (LLMs), but at the cost of increased computational resources. In this work, we identify two key challenges contributing to this inefficiency: $\textit{over-exploration}$ due to redundant states with semantically equivalent content, and $\textit{under-exploration}$ caused by high variance in verifier scoring leading to frequent trajectory switching. To address these issues, we propose FETCH, an e$\textbf{f}$fici$\textbf{e}$nt $\textbf{t}$ree sear$\textbf{ch}$ framework, which is a flexible, plug-and-play system compatible with various tree search algorithms. Our framework mitigates over-exploration by merging semantically similar states using agglomerative clustering of text embeddings obtained from a fine-tuned SimCSE model. To tackle under-exploration, we enhance verifiers by incorporating temporal difference learning with adjusted $\lambda$-returns during training to reduce variance, and employing a verifier ensemble to aggregate scores during inference. Experiments on GSM8K, GSM-Plus, and MATH datasets demonstrate that our methods significantly improve reasoning accuracy and computational efficiency across four different tree search algorithms, paving the way for more practical applications of LLM-based reasoning. The code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07424v2">RomanLens: The Role Of Latent Romanization In Multilinguality In LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 19 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable multilingual generalization despite being predominantly trained on English-centric corpora. A fundamental question arises: how do LLMs achieve such robust multilingual capabilities? We take the case of non-Roman script languages, we investigate the role of Romanization - the representation of non-Roman scripts using Roman characters - as a bridge in multilingual processing. Using mechanistic interpretability techniques, we analyze next-token generation and find that intermediate layers frequently represent target words in Romanized form before transitioning to native script, a phenomenon we term Latent Romanization. Further, through activation patching experiments, we demonstrate that LLMs encode semantic concepts similarly across native and Romanized scripts, suggesting a shared underlying representation. Additionally, for translation into non-Roman script languages, our findings reveal that when the target language is in Romanized form, its representations emerge earlier in the model's layers compared to native script. These insights contribute to a deeper understanding of multilingual representation in LLMs and highlight the implicit role of Romanization in facilitating language transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11155v1">Uncertainty-Aware Search and Value Models: Mitigating Search Scaling Flaws in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Value model-guided search is effective in steering the generation but suffers from scaling flaws: Its superiority diminishes with larger sample sizes, underperforming non-search baselines. This limitation arises from reliability degradation in value models in unseen reasoning paths. To address this, we propose an uncertainty-aware search framework that includes two key components: (1) uncertainty-aware value models that incorporate uncertainty into predictions, and (2) an uncertainty-aware selection process using the proposed efficient Group Thompson Sampling algorithm. Experiments on GSM8K show that our method mitigates search scaling flaws, achieving 90.5% coverage at 16 samples compared to 85.8% for conventional value-guided search. This work establishes the first systematic integration of uncertainty quantification in LLM search paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11149v1">Large Language-Geometry Model: When LLM meets Equivariance</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Accurately predicting 3D structures and dynamics of physical systems is crucial in scientific applications. Existing approaches that rely on geometric Graph Neural Networks (GNNs) effectively enforce $\mathrm{E}(3)$-equivariance, but they often fall in leveraging extensive broader information. While direct application of Large Language Models (LLMs) can incorporate external knowledge, they lack the capability for spatial reasoning with guaranteed equivariance. In this paper, we propose EquiLLM, a novel framework for representing 3D physical systems that seamlessly integrates E(3)-equivariance with LLM capabilities. Specifically, EquiLLM comprises four key components: geometry-aware prompting, an equivariant encoder, an LLM, and an equivariant adaptor. Essentially, the LLM guided by the instructive prompt serves as a sophisticated invariant feature processor, while 3D directional information is exclusively handled by the equivariant encoder and adaptor modules. Experimental results demonstrate that EquiLLM delivers significant improvements over previous methods across molecular dynamics simulation, human motion simulation, and antibody design, highlighting its promising generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00557v3">Learning to Ask: When LLM Agents Meet Unclear Instruction</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Equipped with the capability to call functions, modern large language models (LLMs) can leverage external tools for addressing a range of tasks unattainable through language skills alone. However, the effective execution of these tools relies heavily not just on the advanced capabilities of LLMs but also on precise user instructions, which often cannot be ensured in the real world. To evaluate the performance of LLMs tool-use under imperfect instructions, we meticulously examine the real-world instructions queried from users, analyze the error patterns, and build a challenging tool-use benchmark called Noisy ToolBench (NoisyToolBench). We find that due to the next-token prediction training objective, LLMs tend to arbitrarily generate the missed argument, which may lead to hallucinations and risks. To address this issue, we propose a novel framework, Ask-when-Needed (AwN), which prompts LLMs to ask questions to users whenever they encounter obstacles due to unclear instructions. Moreover, to reduce the manual labor involved in user-LLM interaction and assess LLMs performance in tool utilization from both accuracy and efficiency perspectives, we design an automated evaluation tool named ToolEvaluator. Our experiments demonstrate that the AwN significantly outperforms existing frameworks for tool learning in the NoisyToolBench. We will release all related code and datasets to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11142v1">NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11133v1">MasRouter: Learning to Route LLMs for Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) powered by Large Language Models (LLMs) have been demonstrated to push the boundaries of LLM capabilities, yet they often incur significant costs and face challenges in dynamic LLM selection. Current LLM routing methods effectively reduce overhead in single-agent scenarios by customizing LLM selection for each query, but they overlook the critical decisions regarding collaboration modes and agent roles in MAS. In response to this challenge, we first introduce the problem of Multi-Agent System Routing (MASR), which integrates all components of MAS into a unified routing framework. Toward this goal, we propose MasRouter, the first high-performing, cost-effective, and inductive MASR solution. MasRouter employs collaboration mode determination, role allocation, and LLM routing through a cascaded controller network, progressively constructing a MAS that balances effectiveness and efficiency. Extensive experiments demonstrate that MasRouter is (1) high-performing, achieving a $1.8\%\sim8.2\%$ improvement over the state-of-the-art method on MBPP; (2) economical, reducing overhead by up to $52.07\%$ compared to SOTA methods on HumanEval; and (3) plug-and-play, seamlessly integrating with mainstream MAS frameworks, reducing overhead by $17.21\%\sim28.17\%$ via customized routing. The code is available at https://github.com/yanweiyue/masrouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11127v1">G-Safeguard: A Topology-Guided Security Lens and Treatment on LLM-based Multi-agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based Multi-agent Systems (MAS) have demonstrated remarkable capabilities in various complex tasks, ranging from collaborative problem-solving to autonomous decision-making. However, as these systems become increasingly integrated into critical applications, their vulnerability to adversarial attacks, misinformation propagation, and unintended behaviors have raised significant concerns. To address this challenge, we introduce G-Safeguard, a topology-guided security lens and treatment for robust LLM-MAS, which leverages graph neural networks to detect anomalies on the multi-agent utterance graph and employ topological intervention for attack remediation. Extensive experiments demonstrate that G-Safeguard: (I) exhibits significant effectiveness under various attack strategies, recovering over 40% of the performance for prompt injection; (II) is highly adaptable to diverse LLM backbones and large-scale MAS; (III) can seamlessly combine with mainstream MAS with security guarantees. The code is available at https://github.com/wslong20/G-safeguard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11110v1">Ramp Up NTT in Record Time using GPU-Accelerated Algorithms and LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Homomorphic encryption (HE) is a core building block in privacy-preserving machine learning (PPML), but HE is also widely known as its efficiency bottleneck. Therefore, many GPU-accelerated cryptographic schemes have been proposed to improve the performance of HE. However, these methods often require complex modifications tailored to specific algorithms and are tightly coupled with specific GPU and operating systems. It is interesting to ask how to generally offer more practical GPU-accelerated cryptographic algorithm implementations. Given the powerful code generation capabilities of large language models (LLMs), we aim to explore their potential to automatically generate practical GPU-friendly algorithm code using CPU-friendly code. In this paper, we focus on number theoretic transform (NTT) -- the core mechanism of HE. We first develop and optimize a GPU-friendly NTT (GNTT) family that exploits PyTorch's fast matrix computation and precomputation, achieving an approximately 62x speedup -- a significant boost over existing ones. Then we explore GPU-friendly code generation using various LLMs, including DeepSeek-R1, OpenAI o1 and o3-mini. We discover many interesting findings throughout the process. For instance, somewhat surprisingly, our experiments demonstrate that DeepSeek-R1 significantly outperforms OpenAI o3-mini and o1, but still cannot beat our optimized protocol. The findings provide valuable insights for turbocharging PPML and enhancing code generation capabilities of LLMs. Codes are available at: https://github.com/LMPC-Lab/GenGPUCrypto.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00510v2">Who's the MVP? A Game-Theoretic Evaluation Benchmark for Modular Attribution in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents frameworks often employ modular architectures, incorporating components such as planning, reasoning, action execution, and reflection to tackle complex tasks. However, quantifying the contribution of each module to overall system performance remains a significant challenge, impeding optimization and interpretability. To address this, we introduce CapaBench (Capability-level Assessment Benchmark), an evaluation framework grounded in cooperative game theory's Shapley Value, which systematically measures the marginal impact of individual modules and their interactions within an agent's architecture. By replacing default modules with test variants across all possible combinations, CapaBench provides a principle method for attributing performance contributions. Key contributions include: (1) We are the first to propose a Shapley Value-based methodology for quantifying the contributions of capabilities in LLM agents; (2) Modules with high Shapley Values consistently lead to predictable performance gains when combined, enabling targeted optimization; and (3) We build a multi-round dataset of over 1,500 entries spanning diverse domains and practical task scenarios, enabling comprehensive evaluation of agent capabilities. CapaBench bridges the gap between component-level evaluation and holistic system assessment, providing actionable insights for optimizing modular LLM agents and advancing their deployment in complex, real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11098v1">Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Recent advancements in LLM-based multi-agent (LLM-MA) systems have shown promise, yet significant challenges remain in managing communication and refinement when agents collaborate on complex tasks. In this paper, we propose \textit{Talk Structurally, Act Hierarchically (TalkHier)}, a novel framework that introduces a structured communication protocol for context-rich exchanges and a hierarchical refinement system to address issues such as incorrect outputs, falsehoods, and biases. \textit{TalkHier} surpasses various types of SoTA, including inference scaling model (OpenAI-o1), open-source multi-agent models (e.g., AgentVerse), and majority voting strategies on current LLM and single-agent baselines (e.g., ReAct, GPT4o), across diverse tasks, including open-domain question answering, domain-specific selective questioning, and practical advertisement text generation. These results highlight its potential to set a new standard for LLM-MA systems, paving the way for more effective, adaptable, and collaborative multi-agent frameworks. The code is available https://github.com/sony/talkhier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14326v3">medIKAL: Integrating Knowledge Graphs as Assistants of LLMs for Enhanced Clinical Diagnosis on EMRs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Electronic Medical Records (EMRs), while integral to modern healthcare, present challenges for clinical reasoning and diagnosis due to their complexity and information redundancy. To address this, we proposed medIKAL (Integrating Knowledge Graphs as Assistants of LLMs), a framework that combines Large Language Models (LLMs) with knowledge graphs (KGs) to enhance diagnostic capabilities. medIKAL assigns weighted importance to entities in medical records based on their type, enabling precise localization of candidate diseases within KGs. It innovatively employs a residual network-like approach, allowing initial diagnosis by the LLM to be merged into KG search results. Through a path-based reranking algorithm and a fill-in-the-blank style prompt template, it further refined the diagnostic process. We validated medIKAL's effectiveness through extensive experiments on a newly introduced open-sourced Chinese EMR dataset, demonstrating its potential to improve clinical diagnosis in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19128v2">Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 On going work. Codes are released at https://github.com/zyc140345/FedAMoLE
    </div>
    <details class="paper-abstract">
      Large-scale instruction data is essential for aligning pretrained Large Language Models (LLMs) with human instructions, but may contain sensitive information that hinders its public sharing. Federated Learning (FL) enables collaborative fine-tuning of LLMs without data sharing. However, existing approaches to federated LLM fine-tuning usually adopt a uniform model architecture, making it hard to fit the highly heterogeneous data with varying amounts and formats. To address this, we propose FedAMoLE, a lightweight personalized FL framework that enables data-driven heterogeneous model architectures. This framework features an adaptive mixture of LoRA experts (MoLE) module for aggregating heterogeneous models and a reverse selection-based expert assignment strategy that optimizes model architectures based on data distributions. Experiments across five scenarios show that FedAMoLE improves accuracy by an average of 5.14% compared to existing approaches while obtaining good scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11066v1">CARMA: Enhanced Compositionality in LLMs via Advanced Regularisation and Mutual Information Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 18 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with compositional generalisation, limiting their ability to systematically combine learned components to interpret novel inputs. While architectural modifications, fine-tuning, and data augmentation improve compositionality, they often have limited adaptability, face scalability constraints, or yield diminishing returns on real data. To address this, we propose CARMA, an intervention that enhances the stability and robustness of compositional reasoning in LLMs while preserving fine-tuned performance. CARMA employs mutual information regularisation and layer-wise stability constraints to mitigate feature fragmentation, ensuring structured representations persist across and within layers. We evaluate CARMA on inverse dictionary modelling and sentiment classification, measuring its impact on semantic consistency, performance stability, and robustness to lexical perturbations. Results show that CARMA reduces the variability introduced by fine-tuning, stabilises token representations, and improves compositional reasoning. While its effectiveness varies across architectures, CARMA's key strength lies in reinforcing learned structures rather than introducing new capabilities, making it a scalable auxiliary method. These findings suggest that integrating CARMA with fine-tuning can improve compositional generalisation while maintaining task-specific performance in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11058v1">DreamDDP: Accelerating Data Parallel Distributed LLM Training with Layer-wise Scheduled Partial Synchronization</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      The growth of large language models (LLMs) increases challenges of accelerating distributed training across multiple GPUs in different data centers. Moreover, concerns about data privacy and data exhaustion have heightened interest in geo-distributed data centers. Communication in geo-distributed data parallel training (DDP) with stochastic gradient descent (S-SGD) is the main bottleneck in low-bandwidth environments. Local SGD mitigates communication overhead by reducing synchronization frequency, and recent studies have successfully applied it to geo-distributedly pre-train LLMs. However, we identify that its model synchronization mechanism prevents overlapping communication and computation, which makes the system lose opportunities to overlap communication and computation. To overcome this limitation, we expand the design space of local SGD by layer-wisely decoupling model synchronization. In each iteration, only some layers are synchronized instead of the entire model after a specific number of iterations. Leveraging this methodology, we introduce DreamDDP, a training framework to accelerate low-bandwidth distributed training with three key innovations: (1) partial local SGD with theoretical assurances of convergence rates comparable to S-SGD; (2) overlapping parameter synchronization with computation without extra GPU memory occupation; (3) identifying and exploiting three properties to schedule the communication and computation to reduce the training time based on fine-grained profiling of layer-wise communication and computation time. Empirical evaluations conducted on 32 GPUs using prominent deep learning models, including ResNet-18, ResNet-50, GPT-2, and Llama-2, demonstrate that DreamDDP enhances the convergence properties of Local SGD (and Adam) and achieves speedups ranging from $1.49\times$ to $3.91\times$ over leading baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01743v2">Automating Legal Concept Interpretation with LLMs: Retrieval, Generation, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Legal articles often include vague concepts for adapting to the ever-changing society. Providing detailed interpretations of these concepts is a critical and challenging task even for legal practitioners. It requires meticulous and professional annotations and summarizations by legal experts, which are admittedly time-consuming and expensive to collect at scale. By emulating legal experts' doctrinal method, we introduce a novel framework, ATRIE, using large language models (LLMs) to AuTomatically Retrieve concept-related information, Interpret legal concepts, and Evaluate generated interpretations, eliminating dependence on legal experts. ATRIE comprises a legal concept interpreter and a legal concept interpretation evaluator. The interpreter uses LLMs to retrieve relevant information from judicial precedents and interpret legal concepts. The evaluator uses performance changes on legal concept entailment, a downstream task we propose, as a proxy of interpretation quality. Automatic and multifaceted human evaluations indicate that the quality of our interpretations is comparable to those written by legal experts, with superior comprehensiveness and readability. Although there remains a slight gap in accuracy, it can already assist legal practitioners in improving the efficiency of concept interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11027v1">Diversified Sampling Improves Scaling LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      While increasing training compute has significantly improved the performance of large language models (LLMs), similar gains have not been observed when scaling inference compute. We hypothesize that the primary issue lies in the uniformity of LLM outputs, which leads to inefficient sampling as models repeatedly generate similar but inaccurate responses. Motivated by an intriguing relationship between solution accuracy (Pass@10) and response diversity, we propose DivSampling-a novel and versatile sampling technique designed to enhance the diversity of candidate solutions by introducing prompt perturbations.DivSampling incorporates two categories of perturbations: task-agnostic approaches, which are general and not tailored to any specific task, and task-specific approaches, which are customized based on task content. Our theoretical analysis demonstrates that, under mild assumptions, the error rates of responses generated from diverse prompts are significantly lower compared to those produced by stationary prompts. Comprehensive evaluations across various tasks -including reasoning, mathematics, and code generation - highlight the effectiveness of DivSampling in improving solution accuracy. This scalable and efficient approach offers a new perspective on optimizing test-time inference, addressing limitations in current sampling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11021v1">Leveraging Uncertainty Estimation for Efficient LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) in edge-cloud environments requires an efficient routing strategy to balance cost and response quality. Traditional approaches prioritize either human-preference data or accuracy metrics from benchmark datasets as routing criteria, but these methods suffer from rigidity and subjectivity. Moreover, existing routing frameworks primarily focus on accuracy and cost, neglecting response quality from a human preference perspective. In this work, we propose the Confidence-Driven LLM Router, a novel framework that leverages uncertainty estimation to optimize routing decisions. To comprehensively assess routing performance, we evaluate both system cost efficiency and response quality. In particular, we introduce the novel use of LLM-as-a-Judge to simulate human rating preferences, providing the first systematic assessment of response quality across different routing strategies. Extensive experiments on MT-Bench, GSM8K, and MMLU demonstrate that our approach outperforms state-of-the-art routing methods, achieving superior response quality while maintaining cost efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07261v2">MemHunter: Automated and Verifiable Memorization Detection at Dataset-scale in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to memorize and reproduce content from their training data, raising significant privacy concerns, especially with web-scale datasets. Existing methods for detecting memorization are primarily sample-specific, relying on manually crafted or discretely optimized memory-inducing prompts generated on a per-sample basis, which become impractical for dataset-level detection due to the prohibitive computational cost of iterating through all samples. In real-world scenarios, data owners may need to verify whether a susceptible LLM has memorized their dataset, particularly if the LLM may have collected the data from the web without authorization. To address this, we introduce MemHunter, which trains a memory-inducing LLM and employs hypothesis testing to efficiently detect memorization at the dataset level, without requiring sample-specific memory inducing. Experiments on models like Pythia and Llama demonstrate that MemHunter can extract up to 40% more training data than existing methods under constrained time resources and reduce search time by up to 80% when integrated as a plug-in. Crucially, MemHunter is the first method capable of dataset-level memorization detection, providing a critical tool for assessing privacy risks in LLMs powered by large-scale datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11007v1">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, beyond their large size, make their deployment more challenging during the inference stage. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design a local-cloud LLM inference offloading (LCIO) system, featuring (i) a large-scale cloud LLM that can handle multi-modal data sources and (ii) a lightweight local LLM that can process simple tasks at high speed. LCIO employs resource-constrained reinforcement learning (RCRL) to determine where to make the inference (i.e., local vs. cloud) and which multi-modal data sources to use for each dialogue/task, aiming to maximize the long-term reward (which incorporates response quality, latency, and usage cost) while adhering to resource constraints. We also propose M4A1, a new dataset that accounts for multi-modal, multi-task, multi-dialogue, and multi-LLM characteristics, to investigate the capabilities of LLMs in various practical scenarios. We demonstrate the effectiveness of LCIO compared to baselines, showing significant savings in latency and cost while achieving satisfactory response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10993v1">RoseRAG: Robust Retrieval-augmented Generation with Small-scale LLMs via Margin-aware Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance but face high computational costs and latency, limiting their deployment in resource-constrained settings. In contrast, small-scale LLMs (SLMs) are more efficient yet struggle to capture evolving real-world knowledge. Retrieval-augmented generation (RAG) helps by integrating external knowledge, but imperfect retrieval can introduce distracting noise that misleads SLMs. We propose RoseRAG, a robust RAG framework for SLMs via Margin-aware Preference Optimization. RoseRAG employs multi-turn prompting for detailed reasoning, rejection sampling for high-quality explanations, and contrastive preference selection to refine responses by maximizing the likelihood gap between preferred and non-preferred outputs. By integrating these components into a margin-aware optimization process, RoseRAG robustly enhances the accuracy and reliability of SLMs for RAG applications. Extensive experiments on three open-domain question answering benchmarks indicate that our innovative RoseRAG surpasses state-of-the-art baselines significantly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10978v1">Agentic LLM Framework for Adaptive Decision Discourse</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 24 pages, 4 figures, 1 appendix
    </div>
    <details class="paper-abstract">
      Effective decision-making in complex systems requires synthesizing diverse perspectives to address multifaceted challenges under uncertainty. This study introduces a real-world inspired agentic Large Language Models (LLMs) framework, to simulate and enhance decision discourse-the deliberative process through which actionable strategies are collaboratively developed. Unlike traditional decision-support tools, the framework emphasizes dialogue, trade-off exploration, and the emergent synergies generated by interactions among agents embodying distinct personas. These personas simulate diverse stakeholder roles, each bringing unique priorities, expertise, and value-driven reasoning to the table. The framework incorporates adaptive and self-governing mechanisms, enabling agents to dynamically summon additional expertise and refine their assembly to address evolving challenges. An illustrative hypothetical example focused on extreme flooding in a Midwestern township demonstrates the framework's ability to navigate uncertainty, balance competing priorities, and propose mitigation and adaptation strategies by considering social, economic, and environmental dimensions. Results reveal how the breadth-first exploration of alternatives fosters robust and equitable recommendation pathways. This framework transforms how decisions are approached in high-stakes scenarios and can be incorporated in digital environments. It not only augments decision-makers' capacity to tackle complexity but also sets a foundation for scalable and context-aware AI-driven recommendations. This research explores novel and alternate routes leveraging agentic LLMs for adaptive, collaborative, and equitable recommendation processes, with implications across domains where uncertainty and complexity converge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12701v3">When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to be vulnerable to backdoor attacks, where triggers embedded in poisoned samples can maliciously alter LLMs' behaviors. In this paper, we move beyond attacking LLMs and instead examine backdoor attacks through the novel lens of natural language explanations. Specifically, we leverage LLMs' generative capabilities to produce human-readable explanations for their decisions, enabling direct comparisons between explanations for clean and poisoned samples. Our results show that backdoored models produce coherent explanations for clean inputs but diverse and logically flawed explanations for poisoned data, a pattern consistent across classification and generation tasks for different backdoor attacks. Further analysis reveals key insights into the explanation generation process. At the token level, explanation tokens associated with poisoned samples only appear in the final few transformer layers. At the sentence level, attention dynamics indicate that poisoned inputs shift attention away from the original input context during explanation generation. These findings enhance our understanding of backdoor mechanisms in LLMs and present a promising framework for detecting vulnerabilities through explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04234v2">Functional Homotopy: Smoothing Discrete Optimization via Continuous Parameters for LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Published at ICLR 2025
    </div>
    <details class="paper-abstract">
      Optimization methods are widely employed in deep learning to identify and mitigate undesired model responses. While gradient-based techniques have proven effective for image models, their application to language models is hindered by the discrete nature of the input space. This study introduces a novel optimization approach, termed the \emph{functional homotopy} method, which leverages the functional duality between model training and input generation. By constructing a series of easy-to-hard optimization problems, we iteratively solve these problems using principles derived from established homotopy methods. We apply this approach to jailbreak attack synthesis for large language models (LLMs), achieving a $20\%-30\%$ improvement in success rate over existing methods in circumventing established safe open-source models such as Llama-2 and Llama-3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10953v1">Empirical evaluation of LLMs in predicting fixes of Configuration bugs in Smart Home System</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      This empirical study evaluates the effectiveness of Large Language Models (LLMs) in predicting fixes for configuration bugs in smart home systems. The research analyzes three prominent LLMs - GPT-4, GPT-4o (GPT-4 Turbo), and Claude 3.5 Sonnet - using four distinct prompt designs to assess their ability to identify appropriate fix strategies and generate correct solutions. The study utilized a dataset of 129 debugging issues from the Home Assistant Community, focusing on 21 randomly selected cases for in-depth analysis. Results demonstrate that GPT-4 and Claude 3.5 Sonnet achieved 80\% accuracy in strategy prediction when provided with both bug descriptions and original scripts. GPT-4 exhibited consistent performance across different prompt types, while GPT-4o showed advantages in speed and cost-effectiveness despite slightly lower accuracy. The findings reveal that prompt design significantly impacts model performance, with comprehensive prompts containing both description and original script yielding the best results. This research provides valuable insights for improving automated bug fixing in smart home system configurations and demonstrates the potential of LLMs in addressing configuration-related challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10940v1">CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing many science and engineering fields. However, their huge model sizes impose extremely demanding needs of computational resources in the pre-training stage. Although low-rank factorizations can reduce model parameters, their direct application in LLM pre-training often lead to non-negligible performance loss. To address this fundamental challenge, we introduce CoLA and its memory-efficient implementation, CoLA-M. We leverage the low-rank structure observed widely in model activations, enforcing non-linear transformations between factorized weight matrices to reduce model size, boost model capacity and training efficiency. Experiments on LLaMA models with 60 million to 7 billion parameters show that CoLA reduces the computing cost by $\bf 2\pmb{\times}$ and improves training throughput by $\bf 1.86\pmb{\times}$ while maintaining full-rank level performance. CoLA-M further squeezes memory cost without sacrificing throughput, offering a pre-training approach with collectively superior parameter, computing, and memory efficiency. The LLMs produced are also $\bf 2\pmb{\times}$ smaller, enabling faster inference with lower memory cost on resource-constrained platforms
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08045v2">Break the Checkbox: Challenging Closed-Style Evaluations of Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      A large number of studies rely on closed-style multiple-choice surveys to evaluate cultural alignment in Large Language Models (LLMs). In this work, we challenge this constrained evaluation paradigm and explore more realistic, unconstrained approaches. Using the World Values Survey (WVS) and Hofstede Cultural Dimensions as case studies, we demonstrate that LLMs exhibit stronger cultural alignment in less constrained settings, where responses are not forced. Additionally, we show that even minor changes, such as reordering survey choices, lead to inconsistent outputs, exposing the limitations of closed-style evaluations. Our findings advocate for more robust and flexible evaluation frameworks that focus on specific cultural proxies, encouraging more nuanced and accurate assessments of cultural alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10938v1">PEA: Enhancing LLM Performance on Computational-Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-02-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable capabilities across diverse domains, prompting investigations into their potential as generic reasoning engines. While recent studies have explored inference-time computation to enhance model performance on complex problems, current research lacks a formal framework to characterize the complexity of reasoning tasks. This study introduces the Predicate-Enumeration-Aggregation (PEA) framework, a formal approach to describe and solve a class of important reasoning tasks termed computational reasoning problems. The PEA framework decomposes these problems into predicate and enumeration components, using LLMs to synthesize programs based on specified predicates, enumeration, and aggregation rules. These synthesized programs are then executed to obtain solutions to the computational tasks. We demonstrate the framework's efficacy on benchmark tasks including Boolean satisfiability problems, game of $24$, and planning problems. Empirical evaluation reveals that PEA substantially enhances the performance of underlying models on benchmark computational problems, yielding an average accuracy improvement of approximately $50\%$, coupled with increased efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08661v2">Few-shot LLM Synthetic Data with Distribution Matching</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 10 pages, 5 figures, accepted at www 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance, their ability to perform in-context learning and few-shot language generation has improved significantly. This has spurred using LLMs to produce high-quality synthetic data to enhance the performance of smaller models like online retrievers or weak LLMs. However, LLM-generated synthetic data often differs from the real data in key language attributes (e.g., styles, tones, content proportions, etc.). As a result, mixing these synthetic data directly with real data may distort the original data distribution, potentially hindering performance improvements. To solve this, we introduce SynAlign: a synthetic data generation and filtering framework based on key attribute distribution matching. Before generation, SynAlign employs an uncertainty tracker surrogated by the Gaussian Process model to iteratively select data clusters distinct from selected ones as demonstrations for new data synthesis, facilitating the efficient exploration diversity of the real data. Then, a latent attribute reasoning method is employed: the LLM summarizes linguistic attributes of demonstrations and then synthesizes new data based on them. This approach facilitates synthesizing diverse data with linguistic attributes that appear in real data.After generation, the Maximum Mean Discrepancy is used as the objective function to learn the sampling weight of each synthetic data, ensuring distribution matching with the real data. Our experiments on multiple text prediction tasks show significant performance improvements. We also conducted an online A/B test on an online retriever to demonstrate SynAlign's effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06326v4">Self-Tuning: Instructing LLMs to Effectively Acquire New Knowledge through Self-Teaching</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 35 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle to provide up-to-date information due to their one-time training and the constantly evolving nature of the world. To keep LLMs current, existing approaches typically involve continued pre-training on new documents. However, they frequently face difficulties in extracting stored knowledge. Motivated by the remarkable success of the Feynman Technique in efficient human learning, we introduce Self-Tuning, a learning framework aimed at improving an LLM's ability to effectively acquire new knowledge from unseen raw documents through self-teaching. Specifically, we develop a Self-Teaching strategy that augments the documents with a set of knowledge-intensive tasks created in a self-supervised manner, focusing on three crucial aspects: memorization, comprehension, and self-reflection. Additionally, we introduce three Wiki-Newpages-2023-QA datasets to facilitate an in-depth analysis of an LLM's knowledge acquisition ability concerning memorization, extraction, and reasoning. Extensive experimental results on various models, e.g., Llama2-7B reveal that Self-Tuning consistently exhibits superior performance across all knowledge acquisition tasks and excels in preserving previous knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10638v1">Script&Shift: A Layered Interface Paradigm for Integrating Content Development and Rhetorical Strategy with LLM Writing Assistants</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Good writing is a dynamic process of knowledge transformation, where writers refine and evolve ideas through planning, translating, and reviewing. Generative AI-powered writing tools can enhance this process but may also disrupt the natural flow of writing, such as when using LLMs for complex tasks like restructuring content across different sections or creating smooth transitions. We introduce Script&Shift, a layered interface paradigm designed to minimize these disruptions by aligning writing intents with LLM capabilities to support diverse content development and rhetorical strategies. By bridging envisioning, semantic, and articulatory distances, Script&Shift's interactions allow writers to leverage LLMs for various content development tasks (scripting) and experiment with diverse organization strategies while tailoring their writing for different audiences (shifting). This approach preserves creative control while encouraging divergent and iterative writing. Our evaluation shows that Script&Shift enables writers to creatively and efficiently incorporate LLMs while preserving a natural flow of composition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06394v5">GameArena: Evaluating LLM Reasoning through Live Computer Games</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Evaluating the reasoning abilities of large language models (LLMs) is challenging. Existing benchmarks often depend on static datasets, which are vulnerable to data contamination and may get saturated over time, or on binary live human feedback that conflates reasoning with other abilities. As the most prominent dynamic benchmark, Chatbot Arena evaluates open-ended questions in real-world settings, but lacks the granularity in assessing specific reasoning capabilities. We introduce GameArena, a dynamic benchmark designed to evaluate LLM reasoning capabilities through interactive gameplay with humans. GameArena consists of three games designed to test specific reasoning capabilities (e.g., deductive and inductive reasoning), while keeping participants entertained and engaged. We analyze the gaming data retrospectively to uncover the underlying reasoning processes of LLMs and measure their fine-grained reasoning capabilities. We collect over 2000 game sessions and provide detailed assessments of various reasoning capabilities for five state-of-the-art LLMs. Our user study with 100 participants suggests that GameArena improves user engagement compared to Chatbot Arena. For the first time, GameArena enables the collection of step-by-step LLM reasoning data in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10914v1">LLM-driven Knowledge Distillation for Dynamic Text-Attributed Graphs</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 Accepted at the AI4TS: AI for Time Series Analysis workshop, AAAI 2025
    </div>
    <details class="paper-abstract">
      Dynamic Text-Attributed Graphs (DyTAGs) have numerous real-world applications, e.g. social, collaboration, citation, communication, and review networks. In these networks, nodes and edges often contain text descriptions, and the graph structure can evolve over time. Future link prediction, edge classification, relation generation, and other downstream tasks on DyTAGs require powerful representations that encode structural, temporal, and textual information. Although graph neural networks (GNNs) excel at handling structured data, encoding temporal information within dynamic graphs remains a significant challenge. In this work, we propose LLM-driven Knowledge Distillation for Dynamic Text Attributed Graph (LKD4DyTAG) with temporal encoding to address these challenges. We use a simple, yet effective approach to encode temporal information in edges so that graph convolution can simultaneously capture both temporal and structural information in the hidden representations. To leverage LLM's text processing capabilities for learning richer representations on DyTAGs, we distill knowledge from LLM-driven edge representations (based on a neighborhood's text attributes) into saptio-temporal representations using a lightweight GNN model that encodes temporal and structural information. The objective of knowledge distillation enables the GNN to learn representations that more effectively encode the available structural, temporal, and textual information in DyTAG. We conducted extensive experimentation on six real-world DyTAG datasets to verify the effectiveness of our approach LKD4DyTAG for future link prediction and edge classification task. The results show that our approach significantly improves the performance of downstream tasks compared to the baseline models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03703v2">Human Creativity in the Age of LLMs: Randomized Experiments on Divergent and Convergent Thinking</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 CHI 2025
    </div>
    <details class="paper-abstract">
      Large language models are transforming the creative process by offering unprecedented capabilities to algorithmically generate ideas. While these tools can enhance human creativity when people co-create with them, it's unclear how this will impact unassisted human creativity. We conducted two large pre-registered parallel experiments involving 1,100 participants attempting tasks targeting the two core components of creativity, divergent and convergent thinking. We compare the effects of two forms of large language model (LLM) assistance -- a standard LLM providing direct answers and a coach-like LLM offering guidance -- with a control group receiving no AI assistance, and focus particularly on how all groups perform in a final, unassisted stage. Our findings reveal that while LLM assistance can provide short-term boosts in creativity during assisted tasks, it may inadvertently hinder independent creative performance when users work without assistance, raising concerns about the long-term impact on human creativity and cognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15938v4">RuleR: Improving LLM Controllability by Rule-based Data Recycling</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 NAACL2025 main, Camera-ready
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) still lack delicate controllability over their responses, which is critical to enhancing their performance and the user experience. However, curating supervised fine-tuning (SFT) datasets to improve LLM controllability usually relies on human experts or proprietary LLMs, which requires additional costs. To bridge this gap, we propose Rule-based Data Recycling (RuleR), a data augmentation method incorporating multiple constraints into the original data samples according to predefined rules, which creates new training tasks to consolidate the controllability of LLMs. Instead of creating new data from scratch, RuleR "recycles" existing data by simply applying rule-based edits to their responses and appending the rule-instructions in their original instructions. Experimental results demonstrate RuleR's effectiveness in improving LLM controllability while maintaining general instruction-following capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19959v2">Gender Biases in LLMs: Higher intelligence in LLM does not necessarily solve gender bias and stereotyping</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 1 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are finding applications in all aspects of life, but their susceptibility to biases, particularly gender stereotyping, raises ethical concerns. This study introduces a novel methodology, a persona-based framework, and a unisex name methodology to investigate whether higher-intelligence LLMs reduce such biases. We analyzed 1400 personas generated by two prominent LLMs, revealing that systematic biases persist even in LLMs with higher intelligence and reasoning capabilities. o1 rated males higher in competency (8.1) compared to females (7.9) and non-binary (7.80). The analysis reveals persistent stereotyping across fields like engineering, data, and technology, where the presence of males dominates. Conversely, fields like design, art, and marketing show a stronger presence of females, reinforcing societal notions that associate creativity and communication with females. This paper suggests future directions to mitigate such gender bias, reinforcing the need for further research to reduce biases and create equitable AI models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10871v1">The Representation and Recall of Interwoven Structured Knowledge in LLMs: A Geometric and Layered Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      This study investigates how large language models (LLMs) represent and recall multi-associated attributes across transformer layers. We show that intermediate layers encode factual knowledge by superimposing related attributes in overlapping spaces, along with effective recall even when attributes are not explicitly prompted. In contrast, later layers refine linguistic patterns and progressively separate attribute representations, optimizing task-specific outputs while appropriately narrowing attribute recall. We identify diverse encoding patterns including, for the first time, the observation of 3D spiral structures when exploring information related to the periodic table of elements. Our findings reveal a dynamic transition in attribute representations across layers, contributing to mechanistic interpretability and providing insights for understanding how LLMs handle complex, interrelated knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v1">NitiBench: A Comprehensive Studies of LLM Frameworks Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10858v1">Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Deep iterative chain-of-thought (CoT) reasoning enables LLMs to tackle complex tasks by progressively activating relevant pre-trained knowledge. However, it faces challenges in ensuring continual improvement and determining a stopping criterion. In this paper, we investigate whether the relevant knowledge that contributes directly to solving the given question can be activated from the initial reasoning path, thus circumventing the need for iterative refinement. Our experiments reveal that increasing the diversity of initial reasoning paths can achieve comparable or superior performance, a concept we term \textit{breadth reasoning}. However, existing breadth reasoning approaches, such as self-consistency, offer limited diversity. To address this limitation, we propose a simple yet effective method that enhances reasoning breadth by integrating contextual exploration with reduced sampling randomness. Extensive experiments demonstrate that our approach significantly outperforms deep iterative reasoning. Our code is provided in https://github.com/zongqianwu/breadth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10857v1">Divergent Thoughts toward One Goal: LLM-based Multi-Agent Collaboration System for Electronic Design Automation</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Recently, with the development of tool-calling capabilities in large language models (LLMs), these models have demonstrated significant potential for automating electronic design automation (EDA) flows by interacting with EDA tool APIs via EDA scripts. However, considering the limited understanding of EDA tools, LLMs face challenges in practical scenarios where diverse interfaces of EDA tools exist across different platforms. Additionally, EDA flow automation often involves intricate, long-chain tool-calling processes, increasing the likelihood of errors in intermediate steps. Any errors will lead to the instability and failure of EDA flow automation. To address these challenges, we introduce EDAid, a multi-agent collaboration system where multiple agents harboring divergent thoughts converge towards a common goal, ensuring reliable and successful EDA flow automation. Specifically, each agent is controlled by ChipLlama models, which are expert LLMs fine-tuned for EDA flow automation. Our experiments demonstrate the state-of-the-art (SOTA) performance of our ChipLlama models and validate the effectiveness of our EDAid in the automation of complex EDA flows, showcasing superior performance compared to single-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14708v3">Understanding LLM Embeddings for Regression</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 Published in Transactions on Machine Learning Research (TMLR) 2025. Code can be found in https://github.com/google-research/optformer
    </div>
    <details class="paper-abstract">
      With the rise of large language models (LLMs) for flexibly processing information as strings, a natural application is regression, specifically by preprocessing string representations into LLM embeddings as downstream features for metric prediction. In this paper, we provide one of the first comprehensive investigations into embedding-based regression and demonstrate that LLM embeddings as features can be better for high-dimensional regression tasks than using traditional feature engineering. This regression performance can be explained in part due to LLM embeddings over numeric data inherently preserving Lipschitz continuity over the feature space. Furthermore, we quantify the contribution of different model effects, most notably model size and language understanding, which we find surprisingly do not always improve regression performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10844v1">Be Friendly, Not Friends: How LLM Sycophancy Shapes User Trust</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that large language model (LLM)-powered conversational agents often exhibit `sycophancy', a tendency to adapt their responses to align with user perspectives, even at the expense of factual accuracy. However, users' perceptions of LLM sycophancy and its interplay with other anthropomorphic features (e.g., friendliness) in shaping user trust remains understudied. To bridge this gap, we conducted a 2 (Sycophancy: presence vs. absence) $\times$ 2 (Friendliness: high vs. low) between-subjects experiment ($N = 224$). Our study uncovered, for the first time, the intricate dynamics between LLM sycophancy and friendliness: When an LLM agent already exhibits a friendly demeanor, being sycophantic reduces perceived authenticity, thereby lowering user trust; Conversely, when the agent is less friendly, aligning its responses with user opinions makes it appear more genuine, leading to higher user trust. \add{Our findings entail profound implications for AI persuasion through exploiting human psychological tendencies and highlight the imperative for responsible designs in user-LLM agent interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13144v3">LLMs for Mathematical Modeling: Towards Bridging the Gap between Natural and Mathematical Languages</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 Findings of NAACL2025. Project: https://github.com/FreedomIntelligence/Mamo
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance across various natural language processing tasks, yet their proficiency in mathematical reasoning remains a key challenge. Addressing the gap between natural and mathematical language requires advanced reasoning capabilities, approaching those of Artificial General Intelligence (AGI). However, the evaluation remains challenging, as perfectly representing reality is inherently elusive, and traditional methods like manual or direct comparison of mathematical statements (Ramamonjison et al., 2023) are insufficient for assessing true modeling ability. We propose a process-oriented framework to evaluate LLMs' ability to construct mathematical models, using solvers to compare outputs with ground truth. Introducing Mamo, a benchmark with 1,209 questions covering ordinary differential equations, linear programming, and mixed-integer linear programming, we enable automatic evaluation of modeling accuracy. The results show that existing LLMs struggle with complex mathematical modeling tasks, with larger models demonstrating superior performance, while open-source models remain competitive in simpler cases but still fall short of proprietary models in more challenging problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10768v1">Evaluating improvements on using Large Language Models (LLMs) for property extraction in the Open Research Knowledge Graph (ORKG)</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Current research highlights the great potential of Large Language Models (LLMs) for constructing Scholarly Knowledge Graphs (SKGs). One particularly complex step in this process is relation extraction, aimed at identifying suitable properties to describe the content of research. This study builds directly on previous research of three Open Research Knowledge Graph (ORKG) team members who assessed the readiness of LLMs such as GPT-3.5, Llama 2, and Mistral for property extraction in scientific literature. Given the moderate performance observed, the previous work concluded that fine-tuning is needed to improve these models' alignment with scientific tasks and their emulation of human expertise. Expanding on this prior experiment, this study evaluates the impact of advanced prompt engineering techniques and demonstrates that these techniques can highly significantly enhance the results. Additionally, this study extends the property extraction process to include property matching to existing ORKG properties, which are retrieved via the API. The evaluation reveals that results generated through advanced prompt engineering achieve a higher proportion of matches with ORKG properties, further emphasizing the enhanced alignment achieved. Moreover, this lays the groundwork for addressing challenges such as the inconsistency of ORKG properties, an issue highlighted in prior studies. By assigning unique URIs and using standardized terminology, this work increases the consistency of the properties, fulfilling a crucial aspect of Linked Data and FAIR principles - core commitments of ORKG. This, in turn, significantly enhances the applicability of ORKG content for subsequent tasks such as comparisons of research publications. Finally, the study concludes with recommendations for future improvements in the overall property extraction process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10673v1">Dataset Protection via Watermarked Canaries in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has become an effective method for enhancing large language models (LLMs) with up-to-date knowledge. However, it poses a significant risk of IP infringement, as IP datasets may be incorporated into the knowledge database by malicious Retrieval-Augmented LLMs (RA-LLMs) without authorization. To protect the rights of the dataset owner, an effective dataset membership inference algorithm for RA-LLMs is needed. In this work, we introduce a novel approach to safeguard the ownership of text datasets and effectively detect unauthorized use by the RA-LLMs. Our approach preserves the original data completely unchanged while protecting it by inserting specifically designed canary documents into the IP dataset. These canary documents are created with synthetic content and embedded watermarks to ensure uniqueness, stealthiness, and statistical provability. During the detection process, unauthorized usage is identified by querying the canary documents and analyzing the responses of RA-LLMs for statistical evidence of the embedded watermark. Our experimental results demonstrate high query efficiency, detectability, and stealthiness, along with minimal perturbation to the original dataset, all without compromising the performance of the RAG system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04237v2">Learning to Rewrite: Generalized LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-02-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) present significant risks when used to generate non-factual content and spread disinformation at scale. Detecting such LLM-generated content is crucial, yet current detectors often struggle to generalize in open-world contexts. We introduce Learning2Rewrite, a novel framework for detecting AI-generated text with exceptional generalization to unseen domains. Our method leverages the insight that LLMs inherently modify AI-generated content less than human-written text when tasked with rewriting. By training LLMs to minimize alterations on AI-generated inputs, we amplify this disparity, yielding a more distinguishable and generalizable edit distance across diverse text distributions. Extensive experiments on data from 21 independent domains and four major LLMs (GPT-3.5, GPT-4, Gemini, and Llama-3) demonstrate that our detector outperforms state-of-the-art detection methods by up to 23.04% in AUROC for in-distribution tests, 37.26% for out-of-distribution tests, and 48.66% under adversarial attacks. Our unique training objective ensures better generalizability compared to directly training for classification, when leveraging the same amount of parameters. Our findings suggest that reinforcing LLMs' inherent rewriting tendencies offers a robust and scalable solution for detecting AI-generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10659v1">Pushing up to the Limit of Memory Bandwidth and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA</a></div>
    <div class="paper-meta">
      📅 2025-02-15
      | 💬 Accepted by DATE2025
    </div>
    <details class="paper-abstract">
      The extremely high computational and storage demands of large language models have excluded most edge devices, which were widely used for efficient machine learning, from being viable options. A typical edge device usually only has 4GB of memory capacity and a bandwidth of less than 20GB/s, while a large language model quantized to 4-bit precision with 7B parameters already requires 3.5GB of capacity, and its decoding process is purely bandwidth-bound. In this paper, we aim to explore these limits by proposing a hardware accelerator for large language model (LLM) inference on the Zynq-based KV260 platform, equipped with 4GB of 64-bit 2400Mbps DDR4 memory. We successfully deploy a LLaMA2-7B model, achieving a decoding speed of around 5 token/s, utilizing 93.3% of the memory capacity and reaching 85% decoding speed of the theoretical memory bandwidth limit. To fully reserve the memory capacity for model weights and key-value cache, we develop the system in a bare-metal environment without an operating system. To fully reserve the bandwidth for model weight transfers, we implement a customized dataflow with an operator fusion pipeline and propose a data arrangement format that can maximize the data transaction efficiency. This research marks the first attempt to deploy a 7B level LLM on a standalone embedded field programmable gate array (FPGA) device. It provides key insights into efficient LLM inference on embedded FPGA devices and provides guidelines for future architecture design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20299v2">EACO-RAG: Towards Distributed Tiered LLM Deployment using Edge-Assisted and Collaborative RAG with Adaptive Knowledge Update</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in language tasks, but they require high computing power and rely on static knowledge. To overcome these limitations, Retrieval-Augmented Generation (RAG) incorporates up-to-date external information into LLMs without extensive fine-tuning. Meanwhile, small language models (SLMs) deployed on edge devices offer efficiency and low latency but often struggle with complex reasoning tasks. Unfortunately, current RAG approaches are predominantly based on centralized databases and have not been adapted to address the distinct constraints associated with deploying SLMs in edge environments. To bridge this gap, we propose Edge-Assisted and Collaborative RAG (EACO-RAG), a lightweight framework that leverages distributed edge nodes for adaptive knowledge updates and retrieval. EACO-RAG also employs a hierarchical collaborative gating mechanism to dynamically select among local, edge-assisted, and cloud-based strategies, with a carefully designed algorithm based on Safe Online Bayesian Optimization to maximize the potential performance enhancements. Experimental results demonstrate that EACO-RAG matches the accuracy of cloud-based knowledge graph RAG systems while reducing total costs by up to 84.6% under relaxed delay constraints and by 65.3% under stricter delay requirements. This work represents our initial effort toward achieving a distributed and scalable tiered LLM deployments, with EACO-RAG serving as a promising first step in unlocking the full potential of hybrid edge-cloud intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20856v2">Strada-LLM: Graph LLM for traffic prediction</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 The reviewers decided to reject it. After getting the reviews, we wanted to study more.
    </div>
    <details class="paper-abstract">
      Traffic prediction is a vital component of intelligent transportation systems. By reasoning about traffic patterns in both the spatial and temporal dimensions, accurate and interpretable predictions can be provided. A considerable challenge in traffic prediction lies in handling the diverse data distributions caused by vastly different traffic conditions occurring at different locations. LLMs have been a dominant solution due to their remarkable capacity to adapt to new datasets with very few labeled data samples, i.e., few-shot adaptability. However, existing forecasting techniques mainly focus on extracting local graph information and forming a text-like prompt, leaving LLM- based traffic prediction an open problem. This work presents a probabilistic LLM for traffic forecasting with three highlights. We propose a graph-aware LLM for traffic prediction that considers proximal traffic information. Specifically, by considering the traffic of neighboring nodes as covariates, our model outperforms the corresponding time-series LLM. Furthermore, we adopt a lightweight approach for efficient domain adaptation when facing new data distributions in few-shot fashion. The comparative experiment demonstrates the proposed method outperforms the state-of-the-art LLM-based methods and the traditional GNN- based supervised approaches. Furthermore, Strada-LLM can be easily adapted to different LLM backbones without a noticeable performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14050v3">Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Recent generative large language models (LLMs) show remarkable performance in non-English languages, but when prompted in those languages they tend to express higher harmful social biases and toxicity levels. Prior work has shown that finetuning on specialized datasets can mitigate this behavior, and doing so in English can transfer to other languages. In this work, we investigate the impact of different finetuning methods on the model's bias and toxicity, but also on its ability to produce fluent and diverse text. We reduce biases by finetuning on curated non-harmful text, but find only direct preference optimization to be effective for mitigating toxicity. The mitigation caused by applying these methods in English also transfers to non-English languages. We find evidence that the extent to which transfer takes place can be predicted by the amount of data in a given language present in the model's pretraining data. However, this transfer of bias and toxicity mitigation often comes at the expense of decreased language generation ability in non-English languages, highlighting the importance of developing language-specific bias and toxicity mitigation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10201v1">Prediction hubs are context-informed frequent tokens in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Hubness, the tendency for few points to be among the nearest neighbours of a disproportionate number of other points, commonly arises when applying standard distance measures to high-dimensional data, often negatively impacting distance-based analysis. As autoregressive large language models (LLMs) operate on high-dimensional representations, we ask whether they are also affected by hubness. We first show, theoretically, that the only representation comparison operation performed by LLMs, namely that between context and unembedding vectors to determine continuation probabilities, is not characterized by the concentration of distances phenomenon that typically causes the appeareance of nuisance hubness. We then empirically show that this comparison still leads to a high degree of hubness, but the hubs in this case do not constitute a disturbance. They are rather the result of context-modulated frequent tokens often appearing in the pool of likely candidates for next token prediction. On the other hand, when other distance computations involving LLM representations are performed, we do not have the same theoretical guarantees, and, indeed, we see nuisance hubs appear. In summary, our work highlights, on the one hand, how hubness, while omnipresent in high-dimensional spaces, is not always a negative property that needs to be mitigated, and, on the other hand, it shows that various widely-used LLMs have developed a guessing strategy that consists in constantly assigning a high probability to frequent tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01237v2">Self-Refinement Strategies for LLM-based Product Attribute Value Extraction</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Structured product data, in the form of attribute-value pairs, is essential for e-commerce platforms to support features such as faceted product search and attribute-based product comparison. However, vendors often provide unstructured product descriptions, making attribute value extraction necessary to ensure data consistency and usability. Large language models (LLMs) have demonstrated their potential for product attribute value extraction in few-shot scenarios. Recent research has shown that self-refinement techniques can improve the performance of LLMs on tasks such as code generation and text-to-SQL translation. For other tasks, the application of these techniques has resulted in increased costs due to processing additional tokens, without achieving any improvement in performance. This paper investigates applying two self-refinement techniques (error-based prompt rewriting and self-correction) to the product attribute value extraction task. The self-refinement techniques are evaluated across zero-shot, few-shot in-context learning, and fine-tuning scenarios using GPT-4o. The experiments show that both self-refinement techniques fail to significantly improve the extraction performance while substantially increasing processing costs. For scenarios with development data, fine-tuning yields the highest performance, while the ramp-up costs of fine-tuning are balanced out as the amount of product descriptions increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10197v1">MathConstruct: Challenging LLM Reasoning with Constructive Proofs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) demonstrate impressive performance in mathematics, existing math benchmarks come with significant limitations. Many focus on problems with fixed ground-truth answers, and are often saturated due to problem simplicity or the viability of guessing or memorization. Crucially, they capture only a narrow subset of relevant math problems. To address this research gap, we introduce \mc, a new benchmark of 126 challenging problems sourced from various math competitions, which targets constructive proofs, a widely encountered problem type requiring the construction of mathematical objects with specific properties. These proofs are particularly suitable for LLM evaluation, as solution correctness can be easily verified. Our automated verifiers also enable MathConstruct to generate problem variations, used to evaluate robustness. State-of-the-art LLMs solve only 54% of MathConstruct problems, highlighting its complexity and importance for LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v3">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 Code will be available at https://github.com/iamhankai/Forest-of-Thought
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency.Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14391v2">Context-Aware or Context-Insensitive? Assessing LLMs' Performance in Document-Level Translation</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 9 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly strong contenders in machine translation. In this work, we focus on document-level translation, where some words cannot be translated without context from outside the sentence. Specifically, we investigate the ability of prominent LLMs to utilize the document context during translation through a perturbation analysis (analyzing models' robustness to perturbed and randomized document context) and an attribution analysis (examining the contribution of relevant context to the translation). We conduct an extensive evaluation across nine LLMs from diverse model families and training paradigms, including translation-specialized LLMs, alongside two encoder-decoder transformer baselines. We find that LLMs' improved document-translation performance compared to encoder-decoder models is not reflected in pronoun translation performance. Our analysis highlight the need for context-aware finetuning of LLMs with a focus on relevant parts of the context to improve their reliability for document-level translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10347v2">A Unified Approach to Routing and Cascading for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoretically optimal strategy. Through our analysis, we identify good quality estimators as the critical factor for the success of model selection paradigms. Finally, in our experiments, we show that cascade routing consistently outperforms the individual approaches by a large margin and we analyze quality estimators to determine when routing and/or cascading are useful paradigms for model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v3">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07016v3">Delving into LLM-assisted writing in biomedical publications through excess vocabulary</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 v3: Updating the manuscript to include all PubMed abstracts until the end of 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like ChatGPT can generate and revise text with human-level performance. These models come with clear limitations: they can produce inaccurate information, reinforce existing biases, and be easily misused. Yet, many scientists use them for their scholarly writing. But how wide-spread is such LLM usage in the academic literature? To answer this question for the field of biomedical research, we present an unbiased, large-scale approach: we study vocabulary changes in over 15 million biomedical abstracts from 2010--2024 indexed by PubMed, and show how the appearance of LLMs led to an abrupt increase in the frequency of certain style words. This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs. This lower bound differed across disciplines, countries, and journals, reaching 40% for some subcorpora. We show that LLMs have had an unprecedented impact on scientific writing in biomedical research, surpassing the effect of major world events such as the Covid pandemic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10050v1">A Survey on LLM-powered Agents for Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Recommender systems are essential components of many online platforms, yet traditional approaches still struggle with understanding complex user preferences and providing explainable recommendations. The emergence of Large Language Model (LLM)-powered agents offers a promising approach by enabling natural language interactions and interpretable reasoning, potentially transforming research in recommender systems. This survey provides a systematic review of the emerging applications of LLM-powered agents in recommender systems. We identify and analyze three key paradigms in current research: (1) Recommender-oriented approaches, which leverage intelligent agents to enhance the fundamental recommendation mechanisms; (2) Interaction-oriented approaches, which facilitate dynamic user engagement through natural dialogue and interpretable suggestions; and (3) Simulation-oriented approaches, which employ multi-agent frameworks to model complex user-item interactions and system dynamics. Beyond paradigm categorization, we analyze the architectural foundations of LLM-powered recommendation agents, examining their essential components: profile construction, memory management, strategic planning, and action execution. Our investigation extends to a comprehensive analysis of benchmark datasets and evaluation frameworks in this domain. This systematic examination not only illuminates the current state of LLM-powered agent recommender systems but also charts critical challenges and promising research directions in this transformative field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10038v1">POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      POI representation learning plays a crucial role in handling tasks related to user mobility data. Recent studies have shown that enriching POI representations with multimodal information can significantly enhance their task performance. Previously, the textual information incorporated into POI representations typically involved only POI categories or check-in content, leading to relatively weak textual features in existing methods. In contrast, large language models (LLMs) trained on extensive text data have been found to possess rich textual knowledge. However leveraging such knowledge to enhance POI representation learning presents two key challenges: first, how to extract POI-related knowledge from LLMs effectively, and second, how to integrate the extracted information to enhance POI representations. To address these challenges, we propose POI-Enhancer, a portable framework that leverages LLMs to improve POI representations produced by classic POI learning models. We first design three specialized prompts to extract semantic information from LLMs efficiently. Then, the Dual Feature Alignment module enhances the quality of the extracted information, while the Semantic Feature Fusion module preserves its integrity. The Cross Attention Fusion module then fully adaptively integrates such high-quality information into POI representations and Multi-View Contrastive Learning further injects human-understandable semantic information into these representations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our framework, showing significant improvements across all baseline representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12475v2">RareAgents: Advancing Rare Disease Care through LLM-Empowered Multi-disciplinary Team</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Rare diseases, despite their low individual incidence, collectively impact around 300 million people worldwide due to the vast number of diseases. The involvement of multiple organs and systems, and the shortage of specialized doctors with relevant experience make diagnosing and treating rare diseases more challenging than common diseases. Recently, agents powered by large language models (LLMs) have demonstrated notable applications across various domains. In the medical field, some agent methods have outperformed direct prompts in question-answering tasks from medical examinations. However, current agent frameworks are not well-adapted to real-world clinical scenarios, especially those involving the complex demands of rare diseases. To bridge this gap, we introduce RareAgents, the first LLM-driven multi-disciplinary team framework designed specifically for the complex clinical context of rare diseases. RareAgents integrates advanced Multidisciplinary Team (MDT) coordination, memory mechanisms, and medical tools utilization, leveraging Llama-3.1-8B/70B as the base model. Experimental results show that RareAgents outperforms state-of-the-art domain-specific models, GPT-4o, and current agent frameworks in differential diagnosis and medication recommendation for rare diseases. Furthermore, we contribute a novel rare disease dataset, MIMIC-IV-Ext-Rare, to support further advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09990v1">X-Boundary: Establishing Exact Safety Boundary to Shield LLMs from Multi-Turn Jailbreaks without Compromising Usability</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Despite the rapid development of safety alignment techniques for LLMs, defending against multi-turn jailbreaks is still a challenging task. In this paper, we conduct a comprehensive comparison, revealing that some existing defense methods can improve the robustness of LLMs against multi-turn jailbreaks but compromise usability, i.e., reducing general capabilities or causing the over-refusal problem. From the perspective of mechanism interpretability of LLMs, we discover that these methods fail to establish a boundary that exactly distinguishes safe and harmful feature representations. Therefore, boundary-safe representations close to harmful representations are inevitably disrupted, leading to a decline in usability. To address this issue, we propose X-Boundary to push harmful representations away from boundary-safe representations and obtain an exact distinction boundary. In this way, harmful representations can be precisely erased without disrupting safe ones. Experimental results show that X-Boundary achieves state-of-the-art defense performance against multi-turn jailbreaks, while reducing the over-refusal rate by about 20% and maintaining nearly complete general capability. Furthermore, we theoretically prove and empirically verify that X-Boundary can accelerate the convergence process during training. Please see our code at: https://github.com/AI45Lab/X-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09977v1">LaRA: Benchmarking Retrieval-Augmented Generation and Long-Context LLMs - No Silver Bullet for LC or RAG Routing</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Effectively incorporating external knowledge into Large Language Models (LLMs) is crucial for enhancing their capabilities and addressing real-world needs. Retrieval-Augmented Generation (RAG) offers an effective method for achieving this by retrieving the most relevant fragments into LLMs. However, the advancements in context window size for LLMs offer an alternative approach, raising the question of whether RAG remains necessary for effectively handling external knowledge. Several existing studies provide inconclusive comparisons between RAG and long-context (LC) LLMs, largely due to limitations in the benchmark designs. In this paper, we present LaRA, a novel benchmark specifically designed to rigorously compare RAG and LC LLMs. LaRA encompasses 2,326 test cases across four practical QA task categories and three types of naturally occurring long texts. Through systematic evaluation of seven open-source and four proprietary LLMs, we find that the optimal choice between RAG and LC depends on a complex interplay of factors, including the model's parameter size, long-text capabilities, context length, task type, and the characteristics of the retrieved chunks. Our findings provide actionable guidelines for practitioners to effectively leverage both RAG and LC approaches in developing and deploying LLM applications. Our code and dataset is provided at: \href{https://github.com/likuanppd/LaRA}{\textbf{https://github.com/likuanppd/LaRA}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19324v2">Reward-Guided Speculative Decoding for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. The code is available at https://github.com/BaohaoLiao/RSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09933v1">MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 32 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03824v3">Syntriever: How to Train Your Retriever with Synthetic Data from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-14
      | 💬 the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), Findings, Accepted
    </div>
    <details class="paper-abstract">
      LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{https://github.com/kmswin1/Syntriever}{https://github.com/kmswin1/Syntriever}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v2">AlphaPO - Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Some popular examples of DAAs include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce \textbf{AlphaPO}, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B while achieving 15\% to 50\% relative improvement over DPO on the same models. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10596v1">Post-training an LLM for RAG? Train on Self-Generated Demonstrations</a></div>
    <div class="paper-meta">
      📅 2025-02-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with knowledge intensive NLP tasks, such as answering "Who won the latest World Cup?" because the knowledge they learn during training may be insufficient or outdated. Conditioning generation on retrieved documents -- a technique known as retrieval augmented generation (RAG) -- mitigates these shortcomings by allowing the model to leverage in-context information. Practitioners can improve LLM RAG performance by fine-tuning on retrieval-augmented instructions, but must beware that this can cause undesirable model behaviors like hallucinations. We attribute this degradation to the fact that the training data is likely to be out-of-distribution for the model and may suffer from quality issues, such as misalignment between retrievals and target responses (since retrievals are frequently added post-hoc). We propose a recipe for training RAG-enabled LLMs using self-generated demonstrations, thereby avoiding training on out-of-distribution text and integrating retrievals into the LLM responses. We evaluate our method on knowledge intensive question answering (QA) tasks and show that our method teaches LLMs to properly handle in-context retrievals and abstain from questions it will likely get wrong. Compared to conventional RA-IT methods, our method prevents model degradation in non-RAG settings while exhibiting superior QA performance.
    </details>
</div>
