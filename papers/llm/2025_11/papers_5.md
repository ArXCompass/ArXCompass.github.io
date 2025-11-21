# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09474v2">Surgical AI Copilot: Energy-Based Fourier Gradient Low-Rank Adaptation for Surgical LLM Agent Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Image-guided surgery demands adaptive, real-time decision support, yet static AI models struggle with structured task planning and providing interactive guidance. Large language models (LLMs)-powered agents offer a promising solution by enabling dynamic task planning and predictive decision support. Despite recent advances, the absence of surgical agent datasets and robust parameter-efficient fine-tuning techniques limits the development of LLM agents capable of complex intraoperative reasoning. In this paper, we introduce Surgical AI Copilot, an LLM agent for image-guided pituitary surgery, capable of conversation, planning, and task execution in response to queries involving tasks such as MRI tumor segmentation, endoscope anatomy segmentation, overlaying preoperative imaging with intraoperative views, instrument tracking, and surgical visual question answering (VQA). To enable structured agent planning, we develop the PitAgent dataset, a surgical context-aware planning dataset covering surgical tasks like workflow analysis, instrument localization, anatomical segmentation, and query-based reasoning. Additionally, we propose DEFT-GaLore, a Deterministic Energy-based Fourier Transform (DEFT) gradient projection technique for efficient low-rank adaptation of recent LLMs (e.g., LLaMA 3.2, Qwen 2.5), enabling their use as surgical agent planners. We extensively validate our agent's performance and the proposed adaptation technique against other state-of-the-art low-rank adaptation methods on agent planning and prompt generation tasks, including a zero-shot surgical VQA benchmark, demonstrating the significant potential for truly efficient and scalable surgical LLM agents in real-time operative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09438v1">LLM-Guided Dynamic-UMAP for Personalized Federated Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      We propose a method that uses large language models to assist graph machine learning under personalization and privacy constraints. The approach combines data augmentation for sparse graphs, prompt and instruction tuning to adapt foundation models to graph tasks, and in-context learning to supply few-shot graph reasoning signals. These signals parameterize a Dynamic UMAP manifold of client-specific graph embeddings inside a Bayesian variational objective for personalized federated learning. The method supports node classification and link prediction in low-resource settings and aligns language model latent representations with graph structure via a cross-modal regularizer. We outline a convergence argument for the variational aggregation procedure, describe a differential privacy threat model based on a moments accountant, and present applications to knowledge graph completion, recommendation-style link prediction, and citation and product graphs. We also discuss evaluation considerations for benchmarking LLM-assisted graph machine learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09407v1">CARE-Bench: A Benchmark of Diverse Client Simulations Guided by Expert Principles for Evaluating LLMs in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      The mismatch between the growing demand for psychological counseling and the limited availability of services has motivated research into the application of Large Language Models (LLMs) in this domain. Consequently, there is a need for a robust and unified benchmark to assess the counseling competence of various LLMs. Existing works, however, are limited by unprofessional client simulation, static question-and-answer evaluation formats, and unidimensional metrics. These limitations hinder their effectiveness in assessing a model's comprehensive ability to handle diverse and complex clients. To address this gap, we introduce \textbf{CARE-Bench}, a dynamic and interactive automated benchmark. It is built upon diverse client profiles derived from real-world counseling cases and simulated according to expert guidelines. CARE-Bench provides a multidimensional performance evaluation grounded in established psychological scales. Using CARE-Bench, we evaluate several general-purpose LLMs and specialized counseling models, revealing their current limitations. In collaboration with psychologists, we conduct a detailed analysis of the reasons for LLMs' failures when interacting with clients of different types, which provides directions for developing more comprehensive, universal, and effective counseling models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19254v4">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted by TMLR; UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we outline a versatile framework for closed-book hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we propose a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01166v2">Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22354v2">LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 8 pages (including References). Published at CogSci 2025: https://escholarship.org/uc/item/4932r1hx
    </div>
    <details class="paper-abstract">
      This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09323v1">Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across diverse artificial intelligence tasks, driven by scaling laws that correlate model size and training data with performance improvements. However, this scaling paradigm incurs substantial memory overhead, creating significant challenges for both training and inference. While existing research has primarily addressed parameter and optimizer state memory reduction, activation memory-particularly from feed-forward networks (FFNs)-has become the critical bottleneck, especially when FlashAttention is implemented. In this work, we conduct a detailed memory profiling of LLMs and identify FFN activations as the predominant source to activation memory overhead. Motivated by this, we introduce Mixture-of-Channels (MoC), a novel FFN architecture that selectively activates only the Top-K most relevant channels per token determined by SwiGLU's native gating mechanism. MoC substantially reduces activation memory during pre-training and improves inference efficiency by reducing memory access through partial weight loading into GPU SRAM. Extensive experiments validate that MoC delivers significant memory savings and throughput gains while maintaining competitive model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09586v1">Scaling Environments for LLM Agents in the Era of Learning from Interaction: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 20 pages, 4 figures, SEA Workshop @ NeurIPS 2025
    </div>
    <details class="paper-abstract">
      LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact directly with environments and learn from experience through reinforcement learning. We formalize this iterative process as the Generation-Execution-Feedback (GEF) loop, where environments generate tasks to challenge agents, return observations in response to agents' actions during task execution, and provide evaluative feedback on rollouts for subsequent learning. Under this paradigm, environments function as indispensable producers of experiential data, highlighting the need to scale them toward greater complexity, realism, and interactivity. In this survey, we systematically review representative methods for environment scaling from a pioneering environment-centric perspective and organize them along the stages of the GEF loop, namely task generation, task execution, and feedback. We further analyze benchmarks, implementation strategies, and applications, consolidating fragmented advances and outlining future research directions for agent intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04962v2">Too Good to be Bad: On the Failure of LLMs to Role-Play Villains</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05867v2">MCP-RiskCue: Can LLM Infer Risk Information From MCP Server System Logs?</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: https://github.com/PorUna-byte/MCP-RiskCue
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09223v1">AILINKPREVIEWER: Enhancing Code Reviews with LLM-Powered Link Previews</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Code review is a key practice in software engineering, where developers evaluate code changes to ensure quality and maintainability. Links to issues and external resources are often included in Pull Requests (PRs) to provide additional context, yet they are typically discarded in automated tasks such as PR summarization and code review comment generation. This limits the richness of information available to reviewers and increases cognitive load by forcing context-switching. To address this gap, we present AILINKPREVIEWER, a tool that leverages Large Language Models (LLMs) to generate previews of links in PRs using PR metadata, including titles, descriptions, comments, and link body content. We analyzed 50 engineered GitHub repositories and compared three approaches: Contextual LLM summaries, Non-Contextual LLM summaries, and Metadata-based previews. The results in metrics such as BLEU, BERTScore, and compression ratio show that contextual summaries consistently outperform other methods. However, in a user study with seven participants, most preferred non-contextual summaries, suggesting a trade-off between metric performance and perceived usability. These findings demonstrate the potential of LLM-powered link previews to enhance code review efficiency and to provide richer context for developers and automation in software engineering. The video demo is available at https://www.youtube.com/watch?v=h2qH4RtrB3E, and the tool and its source code can be found at https://github.com/c4rtune/AILinkPreviewer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09148v1">LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) with external tools enables them to execute complex, multi-step tasks. However, tool learning is hampered by the static synthetic data pipelines where data generation and model training are executed as two separate, non-interactive processes. This approach fails to adaptively focus on a model's specific weaknesses and allows noisy labels to persist, degrading training efficiency. We introduce LoopTool, a fully automated, model-aware data evolution framework that closes this loop by tightly integrating data synthesis and model training. LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures. This closed-loop process operates within a cost-effective, open-source ecosystem, eliminating dependence on expensive closed-source APIs. Experiments show that our 8B model trained with LoopTool significantly surpasses its 32B data generator and achieves new state-of-the-art results on the BFCL-v3 and ACEBench benchmarks for its scale. Our work demonstrates that closed-loop, self-refining data pipelines can dramatically enhance the tool-use capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05852v2">Quantifying Edits Decay in Fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 We request the withdrawal of this submission due to technical errors in the manuscript record. Specifically, the author order was set incorrectly, the status was mistakenly marked, and the article has not been published. For these reasons, we kindly ask that the submission be retracted from the system
    </div>
    <details class="paper-abstract">
      Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09133v1">Assessing the Capabilities of LLMs in Humor:A Multi-dimensional Analysis of Oogiri Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Computational humor is a frontier for creating advanced and engaging natural language processing (NLP) applications, such as sophisticated dialogue systems. While previous studies have benchmarked the humor capabilities of Large Language Models (LLMs), they have often relied on single-dimensional evaluations, such as judging whether something is simply ``funny.'' This paper argues that a multifaceted understanding of humor is necessary and addresses this gap by systematically evaluating LLMs through the lens of Oogiri, a form of Japanese improvisational comedy games. To achieve this, we expanded upon existing Oogiri datasets with data from new sources and then augmented the collection with Oogiri responses generated by LLMs. We then manually annotated this expanded collection with 5-point absolute ratings across six dimensions: Novelty, Clarity, Relevance, Intelligence, Empathy, and Overall Funniness. Using this dataset, we assessed the capabilities of state-of-the-art LLMs on two core tasks: their ability to generate creative Oogiri responses and their ability to evaluate the funniness of responses using a six-dimensional evaluation. Our results show that while LLMs can generate responses at a level between low- and mid-tier human performance, they exhibit a notable lack of Empathy. This deficit in Empathy helps explain their failure to replicate human humor assessment. Correlation analyses of human and model evaluation data further reveal a fundamental divergence in evaluation criteria: LLMs prioritize Novelty, whereas humans prioritize Empathy. We release our annotated corpus to the community to pave the way for the development of more emotionally intelligent and sophisticated conversational agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09122v1">Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09105v1">Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 accepted for AAAI 2026 Special Track on AI Alignment
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in real-world systems, making it critical to understand their vulnerabilities. While data poisoning attacks during RLHF/DPO alignment have been studied empirically, their theoretical foundations remain unclear. We investigate the minimum-cost poisoning attack required to steer an LLM's policy toward an attacker's target by flipping preference labels during RLHF/DPO, without altering the compared outputs. We formulate this as a convex optimization problem with linear constraints, deriving lower and upper bounds on the minimum attack cost. As a byproduct of this theoretical analysis, we show that any existing label-flipping attack can be post-processed via our proposed method to reduce the number of label flips required while preserving the intended poisoning effect. Empirical results demonstrate that this cost-minimization post-processing can significantly reduce poisoning costs over baselines, particularly when the reward model's feature dimension is small relative to the dataset size. These findings highlight fundamental vulnerabilities in RLHF/DPO pipelines and provide tools to evaluate their robustness against low-cost poisoning attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14389v2">Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction. This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07061v2">Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets -- IEMOCAP and MELD -- show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09087v1">Tele-LLM-Hub: Building Context-Aware Multi-Agent LLM Systems for Telecom Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      This paper introduces Tele-LLM-Hub, a user friendly low-code solution for rapid prototyping and deployment of context aware multi-agent (MA) Large Language Model (LLM) systems tailored for 5G and beyond. As telecom wireless networks become increasingly complex, intelligent LLM applications must share a domainspecific understanding of network state. We propose TeleMCP, the Telecom Model Context Protocol, to enable structured and context-rich communication between agents in telecom environments. Tele-LLM-Hub actualizes TeleMCP through a low-code interface that supports agent creation, workflow composition, and interaction with software stacks such as srsRAN. Key components include a direct chat interface, a repository of pre-built systems, an Agent Maker leveraging finetuning with our RANSTRUCT framework, and an MA-Maker for composing MA workflows. The goal of Tele-LLM-Hub is to democratize the design of contextaware MA systems and accelerate innovation in next-generation wireless networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10975v2">ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 CCKS 2025 Shared Task Paper
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.18101v3">A Cost-Benefit Analysis of On-Premise Large Language Model Deployment: Breaking Even with Commercial LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming increasingly widespread. Organizations that want to use AI for productivity now face an important decision. They can subscribe to commercial LLM services or deploy models on their own infrastructure. Cloud services from providers such as OpenAI, Anthropic, and Google are attractive because they provide easy access to state-of-the-art models and are easy to scale. However, concerns about data privacy, the difficulty of switching service providers, and long-term operating costs have driven interest in local deployment of open-source models. This paper presents a cost-benefit analysis framework to help organizations determine when on-premise LLM deployment becomes economically viable compared to commercial subscription services. We consider the hardware requirements, operational expenses, and performance benchmarks of the latest open-source models, including Qwen, Llama, Mistral, and etc. Then we compare the total cost of deploying these models locally with the major cloud providers subscription fee. Our findings provide an estimated breakeven point based on usage levels and performance needs. These results give organizations a practical framework for planning their LLM strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07800v1">From Experience to Strategy: Empowering LLM Agents with Trainable Graph Memory</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agent-centric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcement-based weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent's training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents' strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v2">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11773v4">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents -- virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR. Our code is publicly available at https://github.com/ZikangLeng/AgentSense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07322v2">FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      While LLMs have shown great success in financial tasks like stock prediction and question answering, their application in fully automating Equity Research Report generation remains uncharted territory. In this paper, we formulate the Equity Research Report (ERR) Generation task for the first time. To address the data scarcity and the evaluation metrics absence, we present an open-source evaluation benchmark for ERR generation - FinRpt. We frame a Dataset Construction Pipeline that integrates 7 financial data types and produces a high-quality ERR dataset automatically, which could be used for model training and evaluation. We also introduce a comprehensive evaluation system including 11 metrics to assess the generated ERRs. Moreover, we propose a multi-agent framework specifically tailored to address this task, named FinRpt-Gen, and train several LLM-based agents on the proposed datasets using Supervised Fine-Tuning and Reinforcement Learning. Experimental results indicate the data quality and metrics effectiveness of the benchmark FinRpt and the strong performance of FinRpt-Gen, showcasing their potential to drive innovation in the ERR generation field. All code and datasets are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07784v1">Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Multi-agent debate (MAD) has recently emerged as a promising framework for improving the reasoning performance of large language models (LLMs). Yet, whether LLM agents can genuinely engage in deliberative reasoning, beyond simple ensembling or majority voting, remains unclear. We address this question through a controlled study using the Knight--Knave--Spy logic puzzle, which enables precise, step-wise evaluation of debate outcomes and processes under verifiable ground truth. We systematically set up six structural and cognitive factors, including agent team size, composition, confidence visibility, debate order, debate depth, and task difficulty, to disentangle their respective effects on collective reasoning. Our results show that intrinsic reasoning strength and group diversity are the dominant drivers of debate success, while structural parameters such as order or confidence visibility offer limited gains. Beyond outcomes, process-level analyses identify key behavioral patterns: majority pressure suppresses independent correction, effective teams overturn incorrect consensus, and rational, validity-aligned reasoning most strongly predicts improvement. These findings provide valuable insights into how and why LLM debates succeed or fail, offering guidance for designing interpretable and truth-seeking multi-agent reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.03764v2">LLM-based Relevance Assessment for Web-Scale Search Evaluation at Pinterest</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 RecSys 2025 EARL Workshop
    </div>
    <details class="paper-abstract">
      Relevance evaluation plays a crucial role in personalized search systems to ensure that search results align with a user's queries and intent. While human annotation is the traditional method for relevance evaluation, its high cost and long turnaround time limit its scalability. In this work, we present our approach at Pinterest Search to automate relevance evaluation for online experiments using fine-tuned LLMs. We rigorously validate the alignment between LLM-generated judgments and human annotations, demonstrating that LLMs can provide reliable relevance measurement for experiments while greatly improving the evaluation efficiency. Leveraging LLM-based labeling further unlocks the opportunities to expand the query set, optimize sampling design, and efficiently assess a wider range of search experiences at scale. This approach leads to higher-quality relevance metrics and significantly reduces the Minimum Detectable Effect (MDE) in online experiment measurements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14429v3">LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 16 pages, 11 figures, Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Large Language Diffusion Models, or diffusion LLMs, have emerged as a significant focus in NLP research, with substantial effort directed toward understanding their scalability and downstream task performance. However, their long-context capabilities remain unexplored, lacking systematic analysis or methods for context extension. In this work, we present the first systematic investigation comparing the long-context performance of diffusion LLMs and traditional auto-regressive LLMs. We first identify a unique characteristic of diffusion LLMs, unlike auto-regressive LLMs, they maintain remarkably stable perplexity during direct context extrapolation. Moreover, where auto-regressive models fail outright during the Needle-In-A-Haystack task with context exceeding their pretrained length, we discover diffusion LLMs exhibit a distinct local perception phenomenon, enabling successful retrieval from recent context segments. We explain both phenomena through the lens of Rotary Position Embedding (RoPE) scaling theory. Building on these observations, we propose LongLLaDA, a training-free method that integrates LLaDA with the NTK-based RoPE extrapolation. Our results validate that established extrapolation scaling laws remain effective for extending the context windows of diffusion LLMs. Furthermore, we identify long-context tasks where diffusion LLMs outperform auto-regressive LLMs and others where they fall short. Consequently, this study establishes the first length extrapolation method for diffusion LLMs while providing essential theoretical insights and empirical benchmarks critical for advancing future research on long-context diffusion LLMs. The code is available at https://github.com/OpenMOSS/LongLLaDA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01903v2">STAR-1: Safer Alignment of Reasoning LLMs with 1K Data</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      This paper introduces STAR-1, a high-quality, just-1k-scale safety dataset specifically designed for large reasoning models (LRMs) like DeepSeek-R1. Built on three core principles -- diversity, deliberative reasoning, and rigorous filtering -- STAR-1 aims to address the critical needs for safety alignment in LRMs. Specifically, we begin by integrating existing open-source safety datasets from diverse sources. Then, we curate safety policies to generate policy-grounded deliberative reasoning samples. Lastly, we apply a GPT-4o-based safety scoring system to select training examples aligned with best practices. Experimental results show that fine-tuning LRMs with STAR-1 leads to an average 40% improvement in safety performance across four benchmarks, while only incurring a marginal decrease (e.g., an average of 1.1%) in reasoning ability measured across five reasoning tasks. Extensive ablation studies further validate the importance of our design principles in constructing STAR-1 and analyze its efficacy across both LRMs and traditional LLMs. Our project page is https://ucsc-vlaa.github.io/STAR-1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07727v1">LLM-GROP: Visually Grounded Robot Task and Motion Planning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Task planning and motion planning are two of the most important problems in robotics, where task planning methods help robots achieve high-level goals and motion planning methods maintain low-level feasibility. Task and motion planning (TAMP) methods interleave the two processes of task planning and motion planning to ensure goal achievement and motion feasibility. Within the TAMP context, we are concerned with the mobile manipulation (MoMa) of multiple objects, where it is necessary to interleave actions for navigation and manipulation. In particular, we aim to compute where and how each object should be placed given underspecified goals, such as ``set up dinner table with a fork, knife and plate.'' We leverage the rich common sense knowledge from large language models (LLMs), e.g., about how tableware is organized, to facilitate both task-level and motion-level planning. In addition, we use computer vision methods to learn a strategy for selecting base positions to facilitate MoMa behaviors, where the base position corresponds to the robot's ``footprint'' and orientation in its operating space. Altogether, this article provides a principled TAMP framework for MoMa tasks that accounts for common sense about object rearrangement and is adaptive to novel situations that include many objects that need to be moved. We performed quantitative experiments in both real-world settings and simulated environments. We evaluated the success rate and efficiency in completing long-horizon object rearrangement tasks. While the robot completed 84.4\% real-world object rearrangement trials, subjective human evaluations indicated that the robot's performance is still lower than experienced human waiters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07722v1">Critical Confabulation: Can LLMs Hallucinate for Social Good?</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 24 pages, 4 figures, under review
    </div>
    <details class="paper-abstract">
      LLMs hallucinate, yet some confabulations can have social affordances if carefully bounded. We propose critical confabulation (inspired by critical fabulation from literary and social theory), the use of LLM hallucinations to "fill-in-the-gap" for omissions in archives due to social and political inequality, and reconstruct divergent yet evidence-bound narratives for history's "hidden figures". We simulate these gaps with an open-ended narrative cloze task: asking LLMs to generate a masked event in a character-centric timeline sourced from a novel corpus of unpublished texts. We evaluate audited (for data contamination), fully-open models (the OLMo-2 family) and unaudited open-weight and proprietary baselines under a range of prompts designed to elicit controlled and useful hallucinations. Our findings validate LLMs' foundational narrative understanding capabilities to perform critical confabulation, and show how controlled and well-specified hallucinations can support LLM applications for knowledge production without collapsing speculation into a lack of historical accuracy and fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08060v1">From LLMs to Agents: A Comparative Evaluation of LLMs and LLM-based Agents in Security Patch Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The widespread adoption of open-source software (OSS) has accelerated software innovation but also increased security risks due to the rapid propagation of vulnerabilities and silent patch releases. In recent years, large language models (LLMs) and LLM-based agents have demonstrated remarkable capabilities in various software engineering (SE) tasks, enabling them to effectively address software security challenges such as vulnerability detection. However, systematic evaluation of the capabilities of LLMs and LLM-based agents in security patch detection remains limited. To bridge this gap, we conduct a comprehensive evaluation of the performance of LLMs and LLM-based agents for security patch detection. Specifically, we investigate three methods: Plain LLM (a single LLM with a system prompt), Data-Aug LLM (data augmentation based on the Plain LLM), and the ReAct Agent (leveraging the thought-action-observation mechanism). We also evaluate the performance of both commercial and open-source LLMs under these methods and compare these results with those of existing baselines. Furthermore, we analyze the detection performance of these methods across various vulnerability types, and examine the impact of different prompting strategies and context window sizes on the results. Our findings reveal that the Data-Aug LLM achieves the best overall performance, whereas the ReAct Agent demonstrates the lowest false positive rate (FPR). Although baseline methods exhibit strong accuracy, their false positive rates are significantly higher. In contrast, our evaluated methods achieve comparable accuracy while substantially reducing the FPR. These findings provide valuable insights into the practical applications of LLMs and LLM-based agents in security patch detection, highlighting their advantage in maintaining robust performance while minimizing false positive rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25977v3">NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25979v3">AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.21239v5">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the Gram matrix determinant of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring internal access to LLMs. We conduct extensive experiments on both external and internal uncertainty detections, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08798v1">Structured Uncertainty guided Clarification for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      LLM agents extend large language models with tool-calling capabilities, but ambiguous user instructions often lead to incorrect invocations and task failures. We introduce a principled formulation of structured uncertainty over tool-call parameters, modeling joint tool-argument clarification as a POMDP with Expected Value of Perfect Information (EVPI) objective for optimal question selection and aspect-based cost modeling to prevent redundancy. Our SAGE-Agent leverages this structured uncertainty to achieve superior efficiency: increasing coverage on ambiguous tasks by 7-39\% while reducing clarification questions by 1.5-2.7$\times$ compared to strong prompting and uncertainty-based baselines. We present ClarifyBench, the first multi-turn tool-augmented disambiguation benchmark with realistic LLM-based user simulation across diverse domains including document editing, vehicle control, and travel booking. Additionally, we demonstrate that structured uncertainty provides effective training signals for reinforcement learning, boosting When2Call accuracy from 36.5\% to 65.2\% (3B model) and 36.7\% to 62.9\% (7B model) through uncertainty-weighted GRPO training. These results establish structured uncertainty as a principled, efficient approach for tool-augmented agents, improving both task success and interaction efficiency in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14884v3">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 NeurIPS 2025, 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop Selective Head Attention with hardware-efficient, sparsity-aware GPU kernels, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, Qwen, Mistral across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08721v1">Benevolent Dictators? On LLM Agent Behavior in Dictator Games</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 7 pages, 2 figures, v1 init
    </div>
    <details class="paper-abstract">
      In behavioral sciences, experiments such as the ultimatum game are conducted to assess preferences for fairness or self-interest of study participants. In the dictator game, a simplified version of the ultimatum game where only one of two players makes a single decision, the dictator unilaterally decides how to split a fixed sum of money between themselves and the other player. Although recent studies have explored behavioral patterns of AI agents based on Large Language Models (LLMs) instructed to adopt different personas, we question the robustness of these results. In particular, many of these studies overlook the role of the system prompt - the underlying instructions that shape the model's behavior - and do not account for how sensitive results can be to slight changes in prompts. However, a robust baseline is essential when studying highly complex behavioral aspects of LLMs. To overcome previous limitations, we propose the LLM agent behavior study (LLM-ABS) framework to (i) explore how different system prompts influence model behavior, (ii) get more reliable insights into agent preferences by using neutral prompt variations, and (iii) analyze linguistic features in responses to open-ended instructions by LLM agents to better understand the reasoning behind their behavior. We found that agents often exhibit a strong preference for fairness, as well as a significant impact of the system prompt on their behavior. From a linguistic perspective, we identify that models express their responses differently. Although prompt sensitivity remains a persistent challenge, our proposed framework demonstrates a robust foundation for LLM agent behavior studies. Our code artifacts are available at https://github.com/andreaseinwiller/LLM-ABS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08715v1">Bridging Natural Language and ASP: A Hybrid Approach Using LLMs and AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Answer Set Programming (ASP) is a declarative programming paradigm based on logic programming and non-monotonic reasoning. It is a tremendously powerful tool for describing and solving combinatorial problems. Like any other language, ASP requires users to learn how it works and the syntax involved. It is becoming increasingly required for those unfamiliar with programming languages to interact with code. This paper proposes a novel method of translating unconstrained English into ASP programs for logic puzzles using an LLM and Abstract Meaning Representation (AMR) graphs. Everything from ASP rules, facts, and constraints is generated to fully represent and solve the desired problem. Example logic puzzles are used to demonstrate the capabilities of the system. While most current methods rely entirely on an LLM, our system minimizes the role of the LLM only to complete straightforward tasks. The LLM is used to simplify natural language sentences, identify keywords, and generate simple facts. The AMR graphs are then parsed from simplified language and used to generate ASP constraints systematically. The system successfully creates an entire ASP program that solves a combinatorial logic problem. This approach is a significant first step in creating a lighter-weight, explainable system that converts natural language to solve complex logic problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10687v1">Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08575v1">CO2-Meter: A Comprehensive Carbon Footprint Estimator for LLMs on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      LLMs have transformed NLP, yet deploying them on edge devices poses great carbon challenges. Prior estimators remain incomplete, neglecting peripheral energy use, distinct prefill/decode behaviors, and SoC design complexity. This paper presents CO2-Meter, a unified framework for estimating operational and embodied carbon in LLM edge inference. Contributions include: (1) equation-based peripheral energy models and datasets; (2) a GNN-based predictor with phase-specific LLM energy data; (3) a unit-level embodied carbon model for SoC bottleneck analysis; and (4) validation showing superior accuracy over prior methods. Case studies show CO2-Meter's effectiveness in identifying carbon hotspots and guiding sustainable LLM design on edge platforms. Source code: https://github.com/fuzhenxiao/CO2-Meter
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09598v5">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      This paper introduces an infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models in commercial datacenters. The framework combines public API performance data with company-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost and provide a dynamically updated dashboard that visualizes model-level energy, water, and carbon metrics. Results show the most energy-intensive models exceed 29 Wh per long prompt, over 65 times the most efficient systems. Even a 0.42 Wh short query, when scaled to 700M queries/day, aggregates to annual electricity comparable to 35{,}000 U.S. homes, evaporative freshwater equal to the annual drinking needs of 1.2M people, and carbon emissions requiring a Chicago-sized forest to offset. These findings highlight a growing paradox: as AI becomes cheaper and faster, global adoption drives disproportionate resource consumption. Our methodology offers a standardized, empirically grounded basis for sustainability benchmarking and accountability in AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08500v1">SPEAR-MM: Selective Parameter Evaluation and Restoration via Model Merging for Efficient Financial LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) adapted to financial domains often suffer from catastrophic forgetting of general reasoning capabilities essential for customer interactions and complex financial analysis. We introduce Selective Parameter Evaluation and Restoration via Model Merging (SPEAR-MM), a practical framework that preserves critical capabilities while enabling domain adaptation. Our method approximates layer-wise impact on external benchmarks through post-hoc analysis, then selectively freezes or restores transformer layers via spherical interpolation merging. Applied to LLaMA-3.1-8B for financial tasks, SPEAR-MM achieves 91.2% retention of general capabilities versus 69.7% for standard continual pretraining, while maintaining 94% of domain adaptation gains. The approach provides interpretable trade-off control and reduces computational costs by 90% crucial for resource-constrained financial institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08484v1">Patching LLM Like Software: A Lightweight Method for Improving Safety Policy in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      We propose patching for large language models (LLMs) like software versions, a lightweight and modular approach for addressing safety vulnerabilities. While vendors release improved LLM versions, major releases are costly, infrequent, and difficult to tailor to customer needs, leaving released models with known safety gaps. Unlike full-model fine-tuning or major version updates, our method enables rapid remediation by prepending a compact, learnable prefix to an existing model. This "patch" introduces only 0.003% additional parameters, yet reliably steers model behavior toward that of a safer reference model. Across three critical domains (toxicity mitigation, bias reduction, and harmfulness refusal) policy patches achieve safety improvements comparable to next-generation safety-aligned models while preserving fluency. Our results demonstrate that LLMs can be "patched" much like software, offering vendors and practitioners a practical mechanism for distributing scalable, efficient, and composable safety updates between major model releases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08475v1">Designing LLM-based Multi-Agent Systems for Software Engineering Tasks: Quality Attributes, Design Patterns and Rationale</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      As the complexity of Software Engineering (SE) tasks continues to escalate, Multi-Agent Systems (MASs) have emerged as a focal point of research and practice due to their autonomy and scalability. Furthermore, through leveraging the reasoning and planning capabilities of Large Language Models (LLMs), the application of LLM-based MASs in the field of SE is garnering increasing attention. However, there is no dedicated study that systematically explores the design of LLM-based MASs, including the Quality Attributes (QAs) on which the designers mainly focus, the design patterns used by the designers, and the rationale guiding the design of LLM-based MASs for SE tasks. To this end, we conducted a study to identify the QAs that LLM-based MASs for SE tasks focus on, the design patterns used in the MASs, and the design rationale for the MASs. We collected 94 papers on LLM-based MASs for SE tasks as the source. Our study shows that: (1) Code Generation is the most common SE task solved by LLM-based MASs among ten identified SE tasks, (2) Functional Suitability is the QA on which designers of LLM-based MASs pay the most attention, (3) Role-Based Cooperation is the design pattern most frequently employed among 16 patterns used to construct LLM-based MASs, and (4) Improving the Quality of Generated Code is the most common rationale behind the design of LLM-based MASs. Based on the study results, we presented the implications for the design of LLM-based MASs to support SE tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07198v2">Synergy over Discrepancy: A Partition-Based Approach to Multi-Domain LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 20 pages, 5 figures, 21 tables. Accepted at NeurIPS 2025. Corresponding author: Xuan Zhang (xuanzhang2199@gmail.com)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive generalization abilities, yet adapting them effectively across multiple heterogeneous domains remains challenging due to inter-domain interference. To overcome this challenge, we propose a partition-based multi-stage fine-tuning framework designed to exploit inter-domain synergies while minimizing negative transfer. Our approach strategically partitions domains into subsets (stages) by balancing domain discrepancy, synergy, and model capacity constraints. We theoretically analyze the proposed framework and derive novel generalization bounds that justify our partitioning strategy. Extensive empirical evaluations on various language understanding tasks show that our method consistently outperforms state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08394v1">Interaction Dynamics as a Reward Signal for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The alignment of Large Language Models (LLMs) for multi-turn conversations typically relies on reward signals derived from the content of the text. This approach, however, overlooks a rich, complementary source of signal: the dynamics of the interaction itself. This paper introduces TRACE (Trajectory-based Reward for Agent Collaboration Estimation), a novel reward signal derived from the geometric properties of a dialogue's embedding trajectory--a concept we term 'conversational geometry'. Our central finding is that a reward model trained only on these structural signals achieves a pairwise accuracy (68.20%) comparable to a powerful LLM baseline that analyzes the full transcript (70.04%). Furthermore, a hybrid model combining interaction dynamics with textual analysis achieves the highest performance (80.17%), demonstrating their complementary nature. This work provides strong evidence that for interactive settings, how an agent communicates is as powerful a predictor of success as what it says, offering a new, privacy-preserving framework that not only aligns agents but also serves as a diagnostic tool for understanding the distinct interaction patterns that drive successful collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08325v1">AgentPRM: Process Reward Models for LLM Agents via Step-Wise Promise and Progress</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Despite rapid development, large language models (LLMs) still encounter challenges in multi-turn decision-making tasks (i.e., agent tasks) like web shopping and browser navigation, which require making a sequence of intelligent decisions based on environmental feedback. Previous work for LLM agents typically relies on elaborate prompt engineering or fine-tuning with expert trajectories to improve performance. In this work, we take a different perspective: we explore constructing process reward models (PRMs) to evaluate each decision and guide the agent's decision-making process. Unlike LLM reasoning, where each step is scored based on correctness, actions in agent tasks do not have a clear-cut correctness. Instead, they should be evaluated based on their proximity to the goal and the progress they have made. Building on this insight, we propose a re-defined PRM for agent tasks, named AgentPRM, to capture both the interdependence between sequential decisions and their contribution to the final goal. This enables better progress tracking and exploration-exploitation balance. To scalably obtain labeled data for training AgentPRM, we employ a Temporal Difference-based (TD-based) estimation method combined with Generalized Advantage Estimation (GAE), which proves more sample-efficient than prior methods. Extensive experiments across different agentic tasks show that AgentPRM is over $8\times$ more compute-efficient than baselines, and it demonstrates robust improvement when scaling up test-time compute. Moreover, we perform detailed analyses to show how our method works and offer more insights, e.g., applying AgentPRM to the reinforcement learning of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08317v1">Automatic Paper Reviewing with Heterogeneous Graph Reasoning over LLM-Simulated Reviewer-Author Debates</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Existing paper review methods often rely on superficial manuscript features or directly on large language models (LLMs), which are prone to hallucinations, biased scoring, and limited reasoning capabilities. Moreover, these methods often fail to capture the complex argumentative reasoning and negotiation dynamics inherent in reviewer-author interactions. To address these limitations, we propose ReViewGraph (Reviewer-Author Debates Graph Reasoner), a novel framework that performs heterogeneous graph reasoning over LLM-simulated multi-round reviewer-author debates. In our approach, reviewer-author exchanges are simulated through LLM-based multi-agent collaboration. Diverse opinion relations (e.g., acceptance, rejection, clarification, and compromise) are then explicitly extracted and encoded as typed edges within a heterogeneous interaction graph. By applying graph neural networks to reason over these structured debate graphs, ReViewGraph captures fine-grained argumentative dynamics and enables more informed review decisions. Extensive experiments on three datasets demonstrate that ReViewGraph outperforms strong baselines with an average relative improvement of 15.73%, underscoring the value of modeling detailed reviewer-author debate structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08282v1">SRE-Llama -- Fine-Tuned Meta's Llama LLM, Federated Learning, Blockchain and NFT Enabled Site Reliability Engineering(SRE) Platform for Communication and Networking Software Services</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Software services are crucial for reliable communication and networking; therefore, Site Reliability Engineering (SRE) is important to ensure these systems stay reliable and perform well in cloud-native environments. SRE leverages tools like Prometheus and Grafana to monitor system metrics, defining critical Service Level Indicators (SLIs) and Service Level Objectives (SLOs) for maintaining high service standards. However, a significant challenge arises as many developers often lack in-depth understanding of these tools and the intricacies involved in defining appropriate SLIs and SLOs. To bridge this gap, we propose a novel SRE platform, called SRE-Llama, enhanced by Generative-AI, Federated Learning, Blockchain, and Non-Fungible Tokens (NFTs). This platform aims to automate and simplify the process of monitoring, SLI/SLO generation, and alert management, offering ease in accessibility and efficy for developers. The system operates by capturing metrics from cloud-native services and storing them in a time-series database, like Prometheus and Mimir. Utilizing this stored data, our platform employs Federated Learning models to identify the most relevant and impactful SLI metrics for different services and SLOs, addressing concerns around data privacy. Subsequently, fine-tuned Meta's Llama-3 LLM is adopted to intelligently generate SLIs, SLOs, error budgets, and associated alerting mechanisms based on these identified SLI metrics. A unique aspect of our platform is the encoding of generated SLIs and SLOs as NFT objects, which are then stored on a Blockchain. This feature provides immutable record-keeping and facilitates easy verification and auditing of the SRE metrics and objectives. The automation of the proposed platform is governed by the blockchain smart contracts. The proposed SRE-Llama platform prototype has been implemented with a use case featuring a customized Open5GS 5G Core.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.04181v3">Combining LLMs and Knowledge Graphs to Reduce Hallucinations in Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Advancements in natural language processing have revolutionized the way we can interact with digital information systems, such as databases, making them more accessible. However, challenges persist, especially when accuracy is critical, as in the biomedical domain. A key issue is the hallucination problem, where models generate information unsupported by the underlying data, potentially leading to dangerous misinformation. This paper presents a novel approach designed to bridge this gap by combining Large Language Models (LLM) and Knowledge Graphs (KG) to improve the accuracy and reliability of question-answering systems, on the example of a biomedical KG. Built on the LangChain framework, our method incorporates a query checker that ensures the syntactical and semantic validity of LLM-generated queries, which are then used to extract information from a Knowledge Graph, substantially reducing errors like hallucinations. We evaluated the overall performance using a new benchmark dataset of 50 biomedical questions, testing several LLMs, including GPT-4 Turbo and llama3:70b. Our results indicate that while GPT-4 Turbo outperforms other models in generating accurate queries, open-source models like llama3:70b show promise with appropriate prompt engineering. To make this approach accessible, a user-friendly web-based interface has been developed, allowing users to input natural language queries, view generated and corrected Cypher queries, and verify the resulting paths for accuracy. Overall, this hybrid approach effectively addresses common issues such as data gaps and hallucinations, offering a reliable and intuitive solution for question answering systems. The source code for generating the results of this paper and for the user-interface can be found in our Git repository: https://git.zib.de/lpusch/cyphergenkg-gui
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08247v1">ParliaBench: An Evaluation and Benchmarking Framework for LLM-Generated Parliamentary Speech</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Parliamentary speech generation presents specific challenges for large language models beyond standard text generation tasks. Unlike general text generation, parliamentary speeches require not only linguistic quality but also political authenticity and ideological consistency. Current language models lack specialized training for parliamentary contexts, and existing evaluation methods focus on standard NLP metrics rather than political authenticity. To address this, we present ParliaBench, a benchmark for parliamentary speech generation. We constructed a dataset of speeches from UK Parliament to enable systematic model training. We introduce an evaluation framework combining computational metrics with LLM-as-a-judge assessments for measuring generation quality across three dimensions: linguistic quality, semantic coherence, and political authenticity. We propose two novel embedding-based metrics, Political Spectrum Alignment and Party Alignment, to quantify ideological positioning. We fine-tuned five large language models (LLMs), generated 28k speeches, and evaluated them using our framework, comparing baseline and fine-tuned models. Results show that fine-tuning produces statistically significant improvements across the majority of metrics and our novel metrics demonstrate strong discriminative power for political dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08225v1">Benchmarking Educational LLMs with Analytics: A Case Study on Gender Bias in Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As teachers increasingly turn to GenAI in their educational practice, we need robust methods to benchmark large language models (LLMs) for pedagogical purposes. This article presents an embedding-based benchmarking framework to detect bias in LLMs in the context of formative feedback. Using 600 authentic student essays from the AES 2.0 corpus, we constructed controlled counterfactuals along two dimensions: (i) implicit cues via lexicon-based swaps of gendered terms within essays, and (ii) explicit cues via gendered author background in the prompt. We investigated six representative LLMs (i.e. GPT-5 mini, GPT-4o mini, DeepSeek-R1, DeepSeek-R1-Qwen, Gemini 2.5 Pro, Llama-3-8B). We first quantified the response divergence with cosine and Euclidean distances over sentence embeddings, then assessed significance via permutation tests, and finally, visualised structure using dimensionality reduction. In all models, implicit manipulations reliably induced larger semantic shifts for male-female counterfactuals than for female-male. Only the GPT and Llama models showed sensitivity to explicit gender cues. These findings show that even state-of-the-art LLMs exhibit asymmetric semantic responses to gender substitutions, suggesting persistent gender biases in feedback they provide learners. Qualitative analyses further revealed consistent linguistic differences (e.g., more autonomy-supportive feedback under male cues vs. more controlling feedback under female cues). We discuss implications for fairness auditing of pedagogical GenAI, propose reporting standards for counterfactual evaluation in learning analytics, and outline practical guidance for prompt design and deployment to safeguard equitable feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08215v1">Evaluating Gemini LLM in Food Image-Based Recipe and Nutrition Description with EfficientNet-B4 Visual Backbone</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The proliferation of digital food applications necessitates robust methods for automated nutritional analysis and culinary guidance. This paper presents a comprehensive comparative evaluation of a decoupled, multimodal pipeline for food recognition. We evaluate a system integrating a specialized visual backbone (EfficientNet-B4) with a powerful generative large language model (Google's Gemini LLM). The core objective is to evaluate the trade-offs between visual classification accuracy, model efficiency, and the quality of generative output (nutritional data and recipes). We benchmark this pipeline against alternative vision backbones (VGG-16, ResNet-50, YOLOv8) and a lightweight LLM (Gemma). We introduce a formalization for "Semantic Error Propagation" (SEP) to analyze how classification inaccuracies from the visual module cascade into the generative output. Our analysis is grounded in a new Custom Chinese Food Dataset (CCFD) developed to address cultural bias in public datasets. Experimental results demonstrate that while EfficientNet-B4 (89.0\% Top-1 Acc.) provides the best balance of accuracy and efficiency, and Gemini (9.2/10 Factual Accuracy) provides superior generative quality, the system's overall utility is fundamentally bottlenecked by the visual front-end's perceptive accuracy. We conduct a detailed per-class analysis, identifying high semantic similarity as the most critical failure mode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10069v3">ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at NeurIPS 2025 Oral (Thirty-Ninth Conference on Neural Information Processing Systems)
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components -- combined with complex inference pipelines and heterogeneous workloads -- introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. To address this, we introduce Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling; and (3) improves inference efficiency through unified multimodal prefix caching and non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level objectives (SLOs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v20">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08145v1">Still Not There: Can LLMs Outperform Smaller Task-Specific Seq2Seq Models on the Poetry-to-Prose Conversion Task?</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly treated as universal, general-purpose solutions across NLP tasks, particularly in English. But does this assumption hold for low-resource, morphologically rich languages such as Sanskrit? We address this question by comparing instruction-tuned and in-context-prompted LLMs with smaller task-specific encoder-decoder models on the Sanskrit poetry-to-prose conversion task. This task is intrinsically challenging: Sanskrit verse exhibits free word order combined with rigid metrical constraints, and its conversion to canonical prose (anvaya) requires multi-step reasoning involving compound segmentation, dependency resolution, and syntactic linearisation. This makes it an ideal testbed to evaluate whether LLMs can surpass specialised models. For LLMs, we apply instruction fine-tuning on general-purpose models and design in-context learning templates grounded in Paninian grammar and classical commentary heuristics. For task-specific modelling, we fully fine-tune a ByT5-Sanskrit Seq2Seq model. Our experiments show that domain-specific fine-tuning of ByT5-Sanskrit significantly outperforms all instruction-driven LLM approaches. Human evaluation strongly corroborates this result, with scores exhibiting high correlation with Kendall's Tau scores. Additionally, our prompting strategies provide an alternative to fine-tuning when domain-specific verse corpora are unavailable, and the task-specific Seq2Seq model demonstrates robust generalisation on out-of-domain evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08143v1">Relation as a Prior: A Novel Paradigm for LLM-based Document-level Relation Extraction</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated their remarkable capabilities in document understanding. However, recent research reveals that LLMs still exhibit performance gaps in Document-level Relation Extraction (DocRE) as requiring fine-grained comprehension. The commonly adopted "extract entities then predict relations" paradigm in LLM-based methods leads to these gaps due to two main reasons: (1) Numerous unrelated entity pairs introduce noise and interfere with the relation prediction for truly related entity pairs. (2) Although LLMs have identified semantic associations between entities, relation labels beyond the predefined set are still treated as prediction errors. To address these challenges, we propose a novel Relation as a Prior (RelPrior) paradigm for LLM-based DocRE. For challenge (1), RelPrior utilizes binary relation as a prior to extract and determine whether two entities are correlated, thereby filtering out irrelevant entity pairs and reducing prediction noise. For challenge (2), RelPrior utilizes predefined relation as a prior to match entities for triples extraction instead of directly predicting relation. Thus, it avoids misjudgment caused by strict predefined relation labeling. Extensive experiments on two benchmarks demonstrate that RelPrior achieves state-of-the-art performance, surpassing existing LLM-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08128v1">Sentence-Anchored Gist Compression for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      This work investigates context compression for Large Language Models (LLMs) using learned compression tokens to reduce the memory and computational demands of processing long sequences. We demonstrate that pre-trained LLMs can be fine-tuned to compress their context by factors of 2x to 8x without significant performance degradation, as evaluated on both short-context and long-context benchmarks. Furthermore, in experiments on a 3-billion-parameter LLaMA model, our method achieves results on par with alternative compression techniques while attaining higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08098v1">PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at IAS19
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09776v2">Can LLM-Generated Textual Explanations Enhance Model Classification Performance? An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to the 34th International Conference on Artificial Neural Networks (ICANN 2025)
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of Explainable Natural Language Processing (NLP), textual explanations, i.e., human-like rationales, are pivotal for explaining model predictions and enriching datasets with interpretable labels. Traditional approaches rely on human annotation, which is costly, labor-intensive, and impedes scalability. In this work, we present an automated framework that leverages multiple state-of-the-art large language models (LLMs) to generate high-quality textual explanations. We rigorously assess the quality of these LLM-generated explanations using a comprehensive suite of Natural Language Generation (NLG) metrics. Furthermore, we investigate the downstream impact of these explanations on the performance of pre-trained language models (PLMs) and LLMs across natural language inference tasks on two diverse benchmark datasets. Our experiments demonstrate that automated explanations exhibit highly competitive effectiveness compared to human-annotated explanations in improving model performance. Our findings underscore a promising avenue for scalable, automated LLM-based textual explanation generation for extending NLP datasets and enhancing model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08055v1">MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08052v1">Dual-Process Scaffold Reasoning for Enhancing LLM Code Debugging</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent LLMs have demonstrated sophisticated problem-solving capabilities on various benchmarks through advanced reasoning algorithms. However, the key research question of identifying reasoning steps that balance complexity and computational efficiency remains unsolved. Recent research has increasingly drawn upon psychological theories to explore strategies for optimizing cognitive pathways. The LLM's final outputs and intermediate steps are regarded as System 1 and System 2, respectively. However, an in-depth exploration of the System 2 reasoning is still lacking. Therefore, we propose a novel psychologically backed Scaffold Reasoning framework for code debugging, which encompasses the Scaffold Stream, Analytic Stream, and Integration Stream. The construction of reference code within the Scaffold Stream is integrated with the buggy code analysis results produced by the Analytic Stream through the Integration Stream. Our framework achieves an 88.91% pass rate and an average inference time of 5.36 seconds per-problem on DebugBench, outperforming other reasoning approaches across various LLMs in both reasoning accuracy and efficiency. Further analyses elucidate the advantages and limitations of various cognitive pathways across varying problem difficulties and bug types. Our findings also corroborate the alignment of the proposed Scaffold Reasoning framework with human cognitive processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v2">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08012v1">DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Direction-of-Arrival (DOA) estimation is critical in spatial audio and acoustic signal processing, with wide-ranging applications in real-world. Most existing DOA models are trained on synthetic data by convolving clean speech with room impulse responses (RIRs), which limits their generalizability due to constrained acoustic diversity. In this paper, we revisit DOA estimation using a recently introduced dataset constructed with the assistance of large language models (LLMs), which provides more realistic and diverse spatial audio scenes. We benchmark several representative neural-based DOA methods on this dataset and propose LightDOA, a lightweight DOA estimation model based on depthwise separable convolutions, specifically designed for mutil-channel input in varying environments. Experimental results show that LightDOA achieves satisfactory accuracy and robustness across various acoustic scenes while maintaining low computational complexity. This study not only highlights the potential of spatial audio synthesized with the assistance of LLMs in advancing robust and efficient DOA estimation research, but also highlights LightDOA as efficient solution for resource-constrained applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08008v1">Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-view Multi-Label Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07991v1">VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at AAAI 2026 oral
    </div>
    <details class="paper-abstract">
      Competency Questions (CQs) play a crucial role in validating ontology design. While manually crafting CQs can be highly time-consuming and costly for ontology engineers, recent studies have explored the use of large language models (LLMs) to automate this process. However, prior approaches have largely evaluated generated CQs based on their similarity to existing datasets, which often fail to verify semantic pitfalls such as "Misusing allValuesFrom". Since such pitfalls cannot be reliably detected through rule-based methods, we propose a novel dataset and model of Validating Semantic Pitfalls in Ontology (VSPO) for CQ generation specifically designed to verify the semantic pitfalls. To simulate missing and misused axioms, we use LLMs to generate natural language definitions of classes and properties and introduce misalignments between the definitions and the ontology by removing axioms or altering logical operators (e.g., substituting union with intersection). We then fine-tune LLaMA-3.1-8B-Instruct to generate CQs that validate these semantic discrepancies between the provided definitions and the corresponding axioms. The resulting CQs can detect a broader range of modeling errors compared to existing public datasets. Our fine-tuned model demonstrates superior performance over baselines, showing 26% higher precision and 28.2% higher recall than GPT-4.1 in generating CQs for pitfall validation. This research enables automatic generation of TBox-validating CQs using LLMs, significantly reducing manual effort while improving semantic alignment between ontologies and expert knowledge. To the best of our knowledge, this is the first study to target semantic pitfall validation in CQ generation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06209v2">Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07982v1">NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Accurate interpretation of Notices to Airmen (NOTAMs) is critical for aviation safety, yet their condensed and cryptic language poses significant challenges to both manual and automated processing. Existing automated systems are typically limited to shallow parsing, failing to extract the actionable intelligence needed for operational decisions. We formalize the complete interpretation task as deep parsing, a dual-reasoning challenge requiring both dynamic knowledge grounding (linking the NOTAM to evolving real-world aeronautical data) and schema-based inference (applying static domain rules to deduce operational status). To tackle this challenge, we propose NOTAM-Evolve, a self-evolving framework that enables a large language model (LLM) to autonomously master complex NOTAM interpretation. Leveraging a knowledge graph-enhanced retrieval module for data grounding, the framework introduces a closed-loop learning process where the LLM progressively improves from its own outputs, minimizing the need for extensive human-annotated reasoning traces. In conjunction with this framework, we introduce a new benchmark dataset of 10,000 expert-annotated NOTAMs. Our experiments demonstrate that NOTAM-Evolve achieves a 30.4% absolute accuracy improvement over the base LLM, establishing a new state of the art on the task of structured NOTAM interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07971v1">Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to the AAAI Conference on Artificial Intelligence (AAAI-2026)
    </div>
    <details class="paper-abstract">
      We introduce LOREN, a curvature-aware zeroth-order (ZO) optimization method for fine-tuning large language models (LLMs). Existing ZO methods, which estimate gradients via finite differences using random perturbations, often suffer from high variance and suboptimal search directions. Our approach addresses these challenges by: (i) reformulating the problem of gradient preconditioning as that of adaptively estimating an anisotropic perturbation distribution for gradient estimation, (ii) capturing curvature through a low-rank block diagonal preconditioner using the framework of natural evolution strategies, and (iii) applying a REINFORCE leave-one-out (RLOO) gradient estimator to reduce variance. Experiments on standard LLM benchmarks show that our method outperforms state-of-the-art ZO methods by achieving higher accuracy and faster convergence, while cutting peak memory usage by up to 27.3% compared with MeZO-Adam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06778v2">SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026 Extended Version
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility. WARNING: This work may contain content that is offensive and harmful!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07943v1">Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to AAAI 2026. Extended version with full Appendix
    </div>
    <details class="paper-abstract">
      Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at https://github.com/OpenSPG/KAG-Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07910v1">Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 ICLR26 Submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the \textit{Logits-to-Logic} framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01849v2">A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Prompt-based offline methods are commonly used to optimize large language model (LLM) responses, but evaluating these responses is computationally intensive and often fails to accommodate diverse response styles. This study introduces a novel online evaluation framework that employs a multi-agent conversational bandit model to select optimal responses while aligning with user preferences dynamically. To tackle challenges such as high-dimensional features, large response sets, adaptive conversational needs, and multi-device access, we propose MACO, Multi-Agent Conversational Online Learning, which comprises two key components: (1) \texttt{MACO-A}: Executed by local agents, it employs an online elimination mechanism to filter out low-quality responses. (2) \texttt{MACO-S}: Executed by the cloud server, it adaptively adjusts selection strategies based on aggregated preference data. An adaptive preference mechanism triggers asynchronous conversations to enhance alignment efficiency. Theoretical analysis demonstrates that MACO achieves near-optimal regret bounds, matching state-of-the-art performance in various degenerate cases. Extensive experiments utilizing Google and OpenAI text embedding models on the real-world datasets with different response styles, combined with Llama and GPT-4o, show that MACO consistently outperforms baseline methods by at least 8.29\% across varying response set sizes and numbers of agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.05831v3">Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07876v1">LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 14 pages with 7 figures; accepted by the AAAI 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01724v2">ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The NP-hard Dynamic Flexible Job-Shop Scheduling (DFJSP) problem involves real-time events and complex routing. While traditional rules are efficient but rigid, deep learning is opaque and requires feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments demonstrate ReflecSched achieves superior performance, with its best variants attaining an average RPD of 6.04\% and rank of 3.18, significantly outperforming strong traditional and learning-based methods. It also statistically and decisively surpasses direct LLM baselines, securing a 71.35\% Win Rate while being, on average, 15.1\% more token-efficient on Normal-scale problems. Ablation studies attribute this performance to a robust reflection mechanism that leverages high-quality, contrastive experience. This mechanism mitigates key LLM pitfalls like myopic greed, enabling ReflecSched to outperform all evaluated heuristics. Ultimately, the framework's performance is statistically on par with an oracle-like strategy, showcasing its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.07791v9">Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AACL-IJCNLP 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has emerged as a promising alternative to human evaluators across various tasks, yet inherent biases - particularly position bias, the tendency to favor solutions based on their position within the prompt - compromise its reliability. This exploratory study evaluates position bias in LLM judges across pairwise and list-wise comparison settings, introducing three metrics: repetition stability, position consistency, and preference fairness. Our experiments, involving 15 LLM judges across MTBench and DevBench with 22 tasks and approximately 40 solution-generating models, result in over 150,000 evaluation instances. We identify Judge-Level, Candidate-Level, and Task-Level factors contributing to bias. The findings confirm that position bias is not due to random chance and varies significantly across judges and tasks. While position bias is weakly influenced by the length of prompt components, it is strongly affected by the quality gap between solutions. Our agreement and disagreement analysis among judges further provides insights into the distribution of judging difficulty across the dataset, and highlights the potential for dataset modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07865v1">LLM-Powered Fully Automated Chaos Engineering: Towards Enabling Anyone to Build Resilient Software Systems at Low Cost</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at ASE 2025 NIER Track. The code is available at https://github.com/ntt-dkiku/chaos-eater
    </div>
    <details class="paper-abstract">
      Chaos Engineering (CE) is an engineering technique aimed at improving the resilience of distributed systems. It involves intentionally injecting faults into a system to test its resilience, uncover weaknesses, and address them before they cause failures in production. Recent CE tools automate the execution of predefined CE experiments. However, planning such experiments and improving the system based on the experimental results still remain manual. These processes are labor-intensive and require multi-domain expertise. To address these challenges and enable anyone to build resilient systems at low cost, this paper proposes ChaosEater, a system that automates the entire CE cycle with Large Language Models (LLMs). It predefines an agentic workflow according to a systematic CE cycle and assigns subdivided processes within the workflow to LLMs. ChaosEater targets CE for software systems built on Kubernetes. Therefore, the LLMs in ChaosEater complete CE cycles through software engineering tasks, including requirement definition, code generation, testing, and debugging. We evaluate ChaosEater through case studies on small- and large-scale Kubernetes systems. The results demonstrate that it consistently completes reasonable CE cycles with significantly low time and monetary costs. Its cycles are also qualitatively validated by human engineers and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05784v2">DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Please refer to the NeurIPS 2025 submission: https://openreview.net/forum?id=FNuul0hlin The paper has been accepted to the ICML 2025 MUGen Workshop: https://openreview.net/forum?id=ET24oKP23c
    </div>
    <details class="paper-abstract">
      Unlearning in Large Language Models (LLMs) is crucial for protecting private data and removing harmful knowledge. Most existing approaches rely on fine-tuning to balance unlearning efficiency with general language capabilities. However, these methods typically require training or access to retain data, which is often unavailable in real world scenarios. Although these methods can perform well when both forget and retain data are available, few works have demonstrated equivalent capability in more practical, data-limited scenarios. To overcome these limitations, we propose Detect-Reasoning Augmented GeneratiON (DRAGON), a systematic, reasoning-based framework that utilizes in-context chain-of-thought (CoT) instructions to guard deployed LLMs before inference. Instead of modifying the base model, DRAGON leverages the inherent instruction-following ability of LLMs and introduces a lightweight detection module to identify forget-worthy prompts without any retain data. These are then routed through a dedicated CoT guard model to enforce safe and accurate in-context intervention. To robustly evaluate unlearning performance, we introduce novel metrics for unlearning performance and the continual unlearning setting. Extensive experiments across three representative unlearning tasks validate the effectiveness of DRAGON, demonstrating its strong unlearning capability, scalability, and applicability in practical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07842v1">Alignment-Aware Quantization for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 9 pages, 3 figures. Includes 7 pages of supplementary material
    </div>
    <details class="paper-abstract">
      Safety and efficiency are both important factors when deploying large language models(LLMs). LLMs are trained to follow human alignment for safety, and post training quantization(PTQ) is applied afterward for efficiency. However, these two objectives are often in conflict, revealing a fundamental flaw in the conventional PTQ paradigm: quantization can turn into a safety vulnerability if it only aims to achieve low perplexity. Models can demonstrate low perplexity yet exhibit significant degradation in alignment with the safety policy, highlighting that perplexity alone is an insufficient and often misleading proxy for model safety. To address this, we propose Alignment-Aware Quantization(AAQ), a novel approach that integrates Alignment-Preserving Contrastive(APC) loss into the PTQ pipeline. Compared to simple reconstruction loss, ours explicitly preserves alignment by encouraging the quantized model to mimic its safe, instruction-tuned model while diverging from the unaligned, pre-trained counterpart. Our method achieves this robust safety alignment without resorting to specialized safety-focused calibration datasets, highlighting its practical utility and broad applicability. AAQ is compatible with standard PTQ techniques and enables robust 4-bit (W4A4) quantization across diverse model families such as LLaMA, Qwen, and Mistral while maintaining safety where previous methods fail. Our work resolves the critical trade-off between efficiency and safety, paving the way toward LLMs that are both efficient and trustworthy. Anonymized code is available in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07568v1">Procedural Knowledge Improves Agentic LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle when performing agentic tasks without substantial tool support, prom-pt engineering, or fine tuning. Despite research showing that domain-dependent, procedural knowledge can dramatically increase planning efficiency, little work evaluates its potential for improving LLM performance on agentic tasks that may require implicit planning. We formalize, implement, and evaluate an agentic LLM workflow that leverages procedural knowledge in the form of a hierarchical task network (HTN). Empirical results of our implementation show that hand-coded HTNs can dramatically improve LLM performance on agentic tasks, and using HTNs can boost a 20b or 70b parameter LLM to outperform a much larger 120b parameter LLM baseline. Furthermore, LLM-created HTNs improve overall performance, though less so. The results suggest that leveraging expertise--from humans, documents, or LLMs--to curate procedural knowledge will become another important tool for improving LLM workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07698v1">Smart but Costly? Benchmarking LLMs on Functional Accuracy and Energy Efficiency</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      The rapid advancement of AI technologies and their accelerated adoption in software development necessitates a systematic evaluation of their environmental impact alongside functional correctness. While prior studies have examined sustainability in large language models, existing approaches lack systematic frameworks for evaluating accuracy-energy trade-offs in Code Language Models (CLMs). In this paper, we present a framework, BRACE, to benchmark CLMs on a unified scale of energy efficiency and functional correctness (referred to as accuracy). We benchmark 22 state-of-the-art models on code generation and summarization tasks, proposing two rating methods: Concentric Incremental Rating Circles (CIRC) and Observation to Expectation Rating (OTER). CIRC provides deterministic Euclidean-based rankings with static trade-offs that are robust to outliers, and OTER offers trend-aware evaluation with dynamic trade-offs that capture the complex correlation between energy and accuracy, each offering a distinct perspective and addressing the problem in a unique way. These rating methods enable us to rate LLMs on a 1-5 scale reflecting their combined capabilities in terms of energy efficiency and functional correctness. Our analysis reveals models generally perform better in the code summarization tasks as they are not enforced to generate a grammar-based and syntactically correct output. Also, we find that models' size does not have a significant impact on their ratings, indicating that if models utilize their parameters efficiently, they can be ranked higher on these scales. The proposed BRACE framework empowers practitioners to make evidence-based model selections that balance sustainability with task requirements, guiding rating choice -- CIRC for deterministic comparisons or OTER for trend-aware evaluation -- based on deployment priorities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13697v3">RL in Name Only? Analyzing the Structural Assumptions in RL post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based post-training of large language models (LLMs) has recently gained attention, particularly following the release of DeepSeek R1, which applied GRPO for fine-tuning. Amid the growing hype around improved reasoning abilities attributed to RL post-training, we critically examine the formulation and assumptions underlying these methods. We start by highlighting the popular structural assumptions made in modeling LLM training as a Markov Decision Process (MDP), and show how they lead to a degenerate MDP that doesn't quite need the RL/GRPO apparatus. The two critical structural assumptions include (1) making the MDP states be just a concatenation of the actions-with states becoming the context window and the actions becoming the tokens in LLMs and (2) splitting the reward of a state-action trajectory uniformly across the trajectory. Through a comprehensive analysis, we demonstrate that these simplifying assumptions make the approach effectively equivalent to an outcome-driven supervised learning. Our experiments on benchmarks including GSM8K and Countdown using Qwen-2.5 base models show that iterative supervised fine-tuning, incorporating both positive and negative samples, achieves performance comparable to GRPO-based training. We will also argue that the structural assumptions indirectly incentivize the RL to generate longer sequences of intermediate tokens-which in turn feeds into the narrative of "RL generating longer thinking traces." While RL may well be a very useful technique for improving the reasoning abilities of LLMs, our analysis shows that the simplistic structural assumptions made in modeling the underlying MDP render the popular LLM RL frameworks and their interpretations questionable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04715v3">Selection of LLM Fine-Tuning Data based on Orthogonal Rules</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      High-quality training data is critical to the performance of large language models (LLMs). Recent work has explored using LLMs to rate and select data based on a small set of human-designed criteria (rules), but these approaches often rely heavily on heuristics, lack principled metrics for rule evaluation, and generalize poorly to new tasks. We propose a novel rule-based data selection framework that introduces a metric based on the orthogonality of rule score vectors to evaluate and select complementary rules. Our automated pipeline first uses LLMs to generate diverse rules covering multiple aspects of data quality, then rates samples according to these rules and applies the determinantal point process (DPP) to select the most independent rules. These rules are then used to score the full dataset, and high-scoring samples are selected for downstream tasks such as LLM fine-tuning. We evaluate our framework in two experiment setups: (1) alignment with ground-truth ratings and (2) performance of LLMs fine-tuned on the selected data. Experiments across IMDB, Medical, Math, and Code domains demonstrate that our DPP-based rule selection consistently improves both rating accuracy and downstream model performance over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07669v1">Making LLMs Reliable When It Matters Most: A Five-Layer Architecture for High-Stakes Decisions</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 24 pages, 1 figure, 2 tables
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) excel in verifiable domains where outputs can be checked before action but prove less reliable for high-stakes strategic decisions with uncertain outcomes. This gap, driven by mutually reinforcing cognitive biases in both humans and artificial intelligence (AI) systems, threatens the defensibility of valuations and sustainability of investments in the sector. This report describes a framework emerging from systematic qualitative assessment across 7 frontier-grade LLMs and 3 market-facing venture vignettes under time pressure. Detailed prompting specifying decision partnership and explicitly instructing avoidance of sycophancy, confabulation, solution drift, and nihilism achieved initial partnership state but failed to maintain it under operational pressure. Sustaining protective partnership state required an emergent 7-stage calibration sequence, built upon a 4-stage initialization process, within a 5-layer protection architecture enabling bias self-monitoring, human-AI adversarial challenge, partnership state verification, performance degradation detection, and stakeholder protection. Three discoveries resulted: partnership state is achievable through ordered calibration but requires emergent maintenance protocols; reliability degrades when architectural drift and context exhaustion align; and dissolution discipline prevents costly pursuit of fundamentally wrong directions. Cross-model validation revealed systematic performance differences across LLM architectures. This approach demonstrates that human-AI teams can achieve cognitive partnership capable of preventing avoidable regret in high-stakes decisions, addressing return-on-investment expectations that depend on AI systems supporting consequential decision-making without introducing preventable cognitive traps when verification arrives too late.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07659v1">Revisiting NLI: Towards Cost-Effective and Human-Aligned Metrics for Evaluating LLMs in Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Evaluating answers from state-of-the-art large language models (LLMs) is challenging: lexical metrics miss semantic nuances, whereas "LLM-as-Judge" scoring is computationally expensive. We re-evaluate a lightweight alternative -- off-the-shelf Natural Language Inference (NLI) scoring augmented by a simple lexical-match flag and find that this decades-old technique matches GPT-4o's accuracy (89.9%) on long-form QA, while requiring orders-of-magnitude fewer parameters. To test human alignment of these metrics rigorously, we introduce DIVER-QA, a new 3000-sample human-annotated benchmark spanning five QA datasets and five candidate LLMs. Our results highlight that inexpensive NLI-based evaluation remains competitive and offer DIVER-QA as an open resource for future metric research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09205v4">Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at EUSIPCO 2025 - 5 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07641v1">LLMs vs. Traditional Sentiment Tools in Psychology: An Evaluation on Belgian-Dutch Narratives</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Understanding emotional nuances in everyday language is crucial for computational linguistics and emotion research. While traditional lexicon-based tools like LIWC and Pattern have served as foundational instruments, Large Language Models (LLMs) promise enhanced context understanding. We evaluated three Dutch-specific LLMs (ChocoLlama-8B-Instruct, Reynaerde-7B-chat, and GEITje-7B-ultra) against LIWC and Pattern for valence prediction in Flemish, a low-resource language variant. Our dataset comprised approximately 25000 spontaneous textual responses from 102 Dutch-speaking participants, each providing narratives about their current experiences with self-assessed valence ratings (-50 to +50). Surprisingly, despite architectural advancements, the Dutch-tuned LLMs underperformed compared to traditional methods, with Pattern showing superior performance. These findings challenge assumptions about LLM superiority in sentiment analysis tasks and highlight the complexity of capturing emotional valence in spontaneous, real-world narratives. Our results underscore the need for developing culturally and linguistically tailored evaluation frameworks for low-resource language variants, while questioning whether current LLM fine-tuning approaches adequately address the nuanced emotional expressions found in everyday language use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07637v1">Private-RAG: Answering Multiple Queries with LLMs while Keeping Your Data Private</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) enhances large language models (LLMs) by retrieving documents from an external corpus at inference time. When this corpus contains sensitive information, however, unprotected RAG systems are at risk of leaking private information. Prior work has introduced differential privacy (DP) guarantees for RAG, but only in single-query settings, which fall short of realistic usage. In this paper, we study the more practical multi-query setting and propose two DP-RAG algorithms. The first, MURAG, leverages an individual privacy filter so that the accumulated privacy loss only depends on how frequently each document is retrieved rather than the total number of queries. The second, MURAG-ADA, further improves utility by privately releasing query-specific thresholds, enabling more precise selection of relevant documents. Our experiments across multiple LLMs and datasets demonstrate that the proposed methods scale to hundreds of queries within a practical DP budget ($\varepsilon\approx10$), while preserving meaningful utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07628v1">Beyond Correctness: Evaluating and Improving LLM Feedback in Statistical Education</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been proposed as scalable tools to address the gap between the importance of individualized written feedback and the practical challenges of providing it at scale. However, concerns persist regarding the accuracy, depth, and pedagogical value of their feedback responses. The present study investigates the extent to which LLMs can generate feedback that aligns with educational theory and compares techniques to improve their performance. Using mock in-class exam data from two consecutive years of an introductory statistics course at LMU Munich, we evaluated GPT-generated feedback against an established but expanded pedagogical framework. Four enhancement methods were compared in a highly standardized setting, making meaningful comparisons possible: Using a state-of-the-art model, zero-shot prompting, few-shot prompting, and supervised fine-tuning using Low-Rank Adaptation (LoRA). Results show that while all LLM setups reliably provided correctness judgments and explanations, their ability to deliver contextual feedback and suggestions on how students can monitor and regulate their own learning remained limited. Among the tested methods, zero-shot prompting achieved the strongest balance between quality and cost, while fine-tuning required substantially more resources without yielding clear advantages. For educators, this suggests that carefully designed prompts can substantially improve the usefulness of LLM feedback, making it a promising tool, particularly in large introductory courses where students would otherwise receive little or no written feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17120v2">Self-Interpretability: LLMs Can Describe Complex Internal Processes that Drive Their Decisions</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We have only limited understanding of how and why large language models (LLMs) respond in the ways that they do. Their neural networks have proven challenging to interpret, and we are only beginning to tease out the function of individual neurons and circuits within them. However, another path to understanding these systems is to investigate and develop their capacity to explain their own functioning. Here, we show that i) LLMs can accurately describe quantitative features of their own internal processes during certain kinds of decision-making and ii) that it is possible to improve these capabilities through training. To do so, we fine-tuned GPT-4o and GPT-4o-mini to make decisions in a wide variety of complex contexts (e.g., choosing between condos, loans, vacations, etc.) according to randomly-generated, quantitative preferences about how to weigh different attributes (e.g., the relative importance of natural light versus quiet surroundings for condos). We demonstrate that the LLMs can accurately report these preferences (i.e., the weights that they learned to give to different attributes during decision-making). Next, we demonstrate that these LLMs can be fine-tuned to explain their decision-making even more accurately. Finally, we demonstrate that this training generalizes: It improves the ability of the models to accurately explain how they make other complex decisions, not just decisions they have been fine-tuned to make. This work is a step towards training LLMs to accurately and broadly report on their own internal processes -- a possibility that would yield substantial benefits for interpretability, control, and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07585v1">LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 11 pages, 5 figures. To appear in AI4F @ ACM ICAIF '25, November 15-18, 2025, Singapore
    </div>
    <details class="paper-abstract">
      Financial institutions deploy Large Language Models (LLMs) for reconciliations, regulatory reporting, and client communications, but nondeterministic outputs (output drift) undermine auditability and trust. We quantify drift across five model architectures (7B-120B parameters) on regulated financial tasks, revealing a stark inverse relationship: smaller models (Granite-3-8B, Qwen2.5-7B) achieve 100% output consistency at T=0.0, while GPT-OSS-120B exhibits only 12.5% consistency (95% CI: 3.5-36.0%) regardless of configuration (p<0.0001, Fisher's exact test). This finding challenges conventional assumptions that larger models are universally superior for production deployment. Our contributions include: (i) a finance-calibrated deterministic test harness combining greedy decoding (T=0.0), fixed seeds, and SEC 10-K structure-aware retrieval ordering; (ii) task-specific invariant checking for RAG, JSON, and SQL outputs using finance-calibrated materiality thresholds (plus or minus 5%) and SEC citation validation; (iii) a three-tier model classification system enabling risk-appropriate deployment decisions; and (iv) an audit-ready attestation system with dual-provider validation. We evaluated five models (Qwen2.5-7B via Ollama, Granite-3-8B via IBM watsonx.ai, Llama-3.3-70B, Mistral-Medium-2505, and GPT-OSS-120B) across three regulated financial tasks. Across 480 runs (n=16 per condition), structured tasks (SQL) remain stable even at T=0.2, while RAG tasks show drift (25-75%), revealing task-dependent sensitivity. Cross-provider validation confirms deterministic behavior transfers between local and cloud deployments. We map our framework to Financial Stability Board (FSB), Bank for International Settlements (BIS), and Commodity Futures Trading Commission (CFTC) requirements, demonstrating practical pathways for compliance-ready AI deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.09235v3">Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form QA</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Proceedings of the 9th Widening NLP Workshop
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) as chat assistants capable of generating human-like conversations has amplified the need for robust evaluation methods, particularly for open-ended tasks. Conventional metrics such as EM and F1, while useful, are inadequate for capturing the full semantics and contextual depth of such generative outputs. We propose a reference-guided verdict method that automates the evaluation process by leveraging multiple LLMs as judges. Through experiments on free-form question-answering tasks, we demonstrate that combining multiple models improves the reliability and accuracy of evaluations, especially in tasks where a single model may struggle. The results indicate a strong correlation with human evaluations, establishing the proposed method as a reliable alternative to traditional metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20166v2">CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08542v2">CLEV: LLM-Based Evaluation Through Lightweight Efficient Voting for Free-Form Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to AACL 2025
    </div>
    <details class="paper-abstract">
      Evaluating free-form Question Answering (QA) remains a challenge due to its diverse and open-ended nature. Traditional automatic metrics fail to capture semantic equivalence or accommodate the variability of open-ended responses. Leveraging Large Language Models (LLMs) as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. We propose Consensus via Lightweight Efficient Voting (CLEV), which employs two primary LLMs as judges and invokes a third judge only in cases of disagreement. This approach prioritizes evaluation reliability while reducing unnecessary computational demands. Through experiments, including human evaluation, we demonstrate CLEV's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating LLMs on free-form QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07555v1">LLM Optimization Unlocks Real-Time Pairwise Reranking</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07364v1">Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07318v1">When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01934v2">Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 EMNLP 2025 finding
    </div>
    <details class="paper-abstract">
      Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods.
    </details>
</div>
