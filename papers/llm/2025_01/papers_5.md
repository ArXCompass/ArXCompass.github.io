# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07663v1">Enhancing Talent Employment Insights Through Feature Extraction with LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      This paper explores the application of large language models (LLMs) to extract nuanced and complex job features from unstructured job postings. Using a dataset of 1.2 million job postings provided by AdeptID, we developed a robust pipeline to identify and classify variables such as remote work availability, remuneration structures, educational requirements, and work experience preferences. Our methodology combines semantic chunking, retrieval-augmented generation (RAG), and fine-tuning DistilBERT models to overcome the limitations of traditional parsing tools. By leveraging these techniques, we achieved significant improvements in identifying variables often mislabeled or overlooked, such as non-salary-based compensation and inferred remote work categories. We present a comprehensive evaluation of our fine-tuned models and analyze their strengths, limitations, and potential for scaling. This work highlights the promise of LLMs in labor market analytics, providing a foundation for more accurate and actionable insights into job data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07425v1">Enhancing LLM's Ability to Generate More Repository-Aware Unit Tests Through Precise Contextual Information Injection</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Though many learning-based approaches have been proposed for unit test generation and achieved remarkable performance, they still have limitations in relying on task-specific datasets. Recently, Large Language Models (LLMs) guided by prompt engineering have gained attention for their ability to handle a broad range of tasks, including unit test generation. Despite their success, LLMs may exhibit hallucinations when generating unit tests for focal methods or functions due to their lack of awareness regarding the project's global context. These hallucinations may manifest as calls to non-existent methods, as well as incorrect parameters or return values, such as mismatched parameter types or numbers. While many studies have explored the role of context, they often extract fixed patterns of context for different models and focal methods, which may not be suitable for all generation processes (e.g., excessive irrelevant context could lead to redundancy, preventing the model from focusing on essential information). To overcome this limitation, we propose RATester, which enhances the LLM's ability to generate more repository-aware unit tests through global contextual information injection. To equip LLMs with global knowledge similar to that of human testers, we integrate the language server gopls, which provides essential features (e.g., definition lookup) to assist the LLM. When RATester encounters an unfamiliar identifier (e.g., an unfamiliar struct name), it first leverages gopls to fetch relevant definitions and documentation comments, and then uses this global knowledge to guide the LLM. By utilizing gopls, RATester enriches the LLM's knowledge of the project's global context, thereby reducing hallucinations during unit test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11547v2">Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing SLMs with Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Accepted as Main Conference Paper at COLING 2025
    </div>
    <details class="paper-abstract">
      In this paper, we evaluate the creative fiction writing abilities of a fine-tuned small language model (SLM), BART-large, and compare its performance to human writers and two large language models (LLMs): GPT-3.5 and GPT-4o. Our evaluation consists of two experiments: (i) a human study in which 68 participants rated short stories from humans and the SLM on grammaticality, relevance, creativity, and attractiveness, and (ii) a qualitative linguistic analysis examining the textual characteristics of stories produced by each model. In the first experiment, BART-large outscored average human writers overall (2.11 vs. 1.85), a 14% relative improvement, though the slight human advantage in creativity was not statistically significant. In the second experiment, qualitative analysis showed that while GPT-4o demonstrated near-perfect coherence and used less cliche phrases, it tended to produce more predictable language, with only 3% of its synopses featuring surprising associations (compared to 15% for BART). These findings highlight how model size and fine-tuning influence the balance between creativity, fluency, and coherence in creative writing tasks, and demonstrate that smaller models can, in certain contexts, rival both humans and larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09536v2">Galapagos: Automated N-Version Programming with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      N-Version Programming is a well-known methodology for developing fault-tolerant systems. It achieves fault detection and correction at runtime by adding diverse redundancy into programs, minimizing fault mode overlap between redundant program variants. In this work, we propose the automated generation of program variants using large language models. We design, develop and evaluate Gal\'apagos: a tool for generating program variants using LLMs, validating their correctness and equivalence, and using them to assemble N-Version binaries. We evaluate Gal\'apagos by creating N-Version components of real-world C code. Our original results show that Gal\'apagos can produce program variants that are proven to be functionally equivalent, even when the variants are written in a different programming language. Our systematic diversity measurement indicates that functionally equivalent variants produced by Gal\'apagos, are statically different after compilation, and present diverging internal behavior at runtime. We demonstrate that the variants produced by Gal\'apagos can protect C code against real miscompilation bugs which affect the Clang compiler. Overall, our paper shows that producing N-Version software can be drastically automated by advanced usage of practical formal verification and generative language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07314v1">FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 11 pages, 4 figures, 4 tables. To be published in NoDaLiDa/Baltic-HLT 2025 proceedings
    </div>
    <details class="paper-abstract">
      Data quality is crucial for training Large Language Models (LLMs). Traditional heuristic filters often miss low-quality text or mistakenly remove valuable content. In this paper, we introduce an LLM-based line-level filtering method to enhance training data quality. We use GPT-4o mini to label a 20,000-document sample from FineWeb at the line level, allowing the model to create descriptive labels for low-quality lines. These labels are grouped into nine main categories, and we train a DeBERTa-v3 classifier to scale the filtering to a 10B-token subset of FineWeb. To test the impact of our filtering, we train GPT-2 models on both the original and the filtered datasets. The results show that models trained on the filtered data achieve higher accuracy on the HellaSwag benchmark and reach their performance targets faster, even with up to 25\% less data. This demonstrates that LLM-based line-level filtering can significantly improve data quality and training efficiency for LLMs. We release our quality-annotated dataset, FinerWeb-10BT, and the codebase to support further work in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07288v1">LLM-Net: Democratizing LLMs-as-a-Service through Blockchain-based Expert Networks</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      The centralization of Large Language Models (LLMs) development has created significant barriers to AI advancement, limiting the democratization of these powerful technologies. This centralization, coupled with the scarcity of high-quality training data and mounting complexity of maintaining comprehensive expertise across rapidly expanding knowledge domains, poses critical challenges to the continued growth of LLMs. While solutions like Retrieval-Augmented Generation (RAG) offer potential remedies, maintaining up-to-date expert knowledge across diverse domains remains a significant challenge, particularly given the exponential growth of specialized information. This paper introduces LLMs Networks (LLM-Net), a blockchain-based framework that democratizes LLMs-as-a-Service through a decentralized network of specialized LLM providers. By leveraging collective computational resources and distributed domain expertise, LLM-Net incorporates fine-tuned expert models for various specific domains, ensuring sustained knowledge growth while maintaining service quality through collaborative prompting mechanisms. The framework's robust design includes blockchain technology for transparent transaction and performance validation, establishing an immutable record of service delivery. Our simulation, built on top of state-of-the-art LLMs such as Claude 3.5 Sonnet, Llama 3.1, Grok-2, and GPT-4o, validates the effectiveness of the reputation-based mechanism in maintaining service quality by selecting high-performing respondents (LLM providers). Thereby it demonstrates the potential of LLM-Net to sustain AI advancement through the integration of decentralized expertise and blockchain-based accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v1">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 14 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12094v2">Are LLMs Good Cryptic Crossword Solvers?</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Cryptic crosswords are puzzles that rely not only on general knowledge but also on the solver's ability to manipulate language on different levels and deal with various types of wordplay. Previous research suggests that solving such puzzles is a challenge even for modern NLP models. However, the abilities of large language models (LLMs) have not yet been tested on this task. In this paper, we establish the benchmark results for three popular LLMs -- LLaMA2, Mistral, and ChatGPT -- showing that their performance on this task is still far from that of humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07237v1">Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance across a range of natural language processing tasks. However, their vast number of parameters introduces significant memory challenges during training, particularly when using memory-intensive optimizers like Adam. Existing memory-efficient algorithms often rely on techniques such as singular value decomposition projection or weight freezing. While these approaches help alleviate memory constraints, they generally produce suboptimal results compared to full-rank updates. In this paper, we investigate the memory-efficient method beyond low-rank training, proposing a novel solution called Gradient Wavelet Transform (GWT), which applies wavelet transforms to gradients in order to significantly reduce the memory requirements for maintaining optimizer states. We demonstrate that GWT can be seamlessly integrated with memory-intensive optimizers, enabling efficient training without sacrificing performance. Through extensive experiments on both pre-training and fine-tuning tasks, we show that GWT achieves state-of-the-art performance compared with advanced memory-efficient optimizers and full-rank approaches in terms of both memory usage and training performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07224v1">Touched by ChatGPT: Using an LLM to Drive Affective Tactile Interaction</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Touch is a fundamental aspect of emotion-rich communication, playing a vital role in human interaction and offering significant potential in human-robot interaction. Previous research has demonstrated that a sparse representation of human touch can effectively convey social tactile signals. However, advances in human-robot tactile interaction remain limited, as many humanoid robots possess simplistic capabilities, such as only opening and closing their hands, restricting nuanced tactile expressions. In this study, we explore how a robot can use sparse representations of tactile vibrations to convey emotions to a person. To achieve this, we developed a wearable sleeve integrated with a 5x5 grid of vibration motors, enabling the robot to communicate diverse tactile emotions and gestures. Using chain prompts within a Large Language Model (LLM), we generated distinct 10-second vibration patterns corresponding to 10 emotions (e.g., happiness, sadness, fear) and 6 touch gestures (e.g., pat, rub, tap). Participants (N = 32) then rated each vibration stimulus based on perceived valence and arousal. People are accurate at recognising intended emotions, a result which aligns with earlier findings. These results highlight the LLM's ability to generate emotional haptic data and effectively convey emotions through tactile signals. By translating complex emotional and tactile expressions into vibratory patterns, this research demonstrates how LLMs can enhance physical interaction between humans and robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18873v3">LayoutCopilot: An LLM-powered Multi-agent Collaborative Framework for Interactive Analog Layout Design</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 8pages, 8figures
    </div>
    <details class="paper-abstract">
      Analog layout design heavily involves interactive processes between humans and design tools. Electronic Design Automation (EDA) tools for this task are usually designed to use scripting commands or visualized buttons for manipulation, especially for interactive automation functionalities, which have a steep learning curve and cumbersome user experience, making a notable barrier to designers' adoption. Aiming to address such a usability issue, this paper introduces LayoutCopilot, a pioneering multi-agent collaborative framework powered by Large Language Models (LLMs) for interactive analog layout design. LayoutCopilot simplifies human-tool interaction by converting natural language instructions into executable script commands, and it interprets high-level design intents into actionable suggestions, significantly streamlining the design process. Experimental results demonstrate the flexibility, efficiency, and accessibility of LayoutCopilot in handling real-world analog designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14047v3">An empirical study of LLaMA3 quantization: from LLMs to MLLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      The LLaMA family, a collection of foundation language models ranging from 7B to 65B parameters, has become one of the most powerful open-source large language models (LLMs) and the popular LLM backbone of multi-modal large language models (MLLMs), widely used in computer vision and natural language understanding tasks. In particular, LLaMA3 models have recently been released and have achieved impressive performance in various domains with super-large scale pre-training on over 15T tokens of data. Given the wide application of low-bit quantization for LLMs in resource-constrained scenarios, we explore LLaMA3's capabilities when quantized to low bit-width. This exploration can potentially provide new insights and challenges for the low-bit quantization of LLaMA3 and other future LLMs, especially in addressing performance degradation issues that suffer in LLM compression. Specifically, we comprehensively evaluate the 10 existing post-training quantization and LoRA fine-tuning (LoRA-FT) methods of LLaMA3 on 1-8 bits and various datasets to reveal the low-bit quantization performance of LLaMA3. To uncover the capabilities of low-bit quantized MLLM, we assessed the performance of the LLaMA3-based LLaVA-Next-8B model under 2-4 ultra-low bits with post-training quantization methods. Our experimental results indicate that LLaMA3 still suffers from non-negligible degradation in linguistic and visual contexts, particularly under ultra-low bit widths. This highlights the significant performance gap at low bit-width that needs to be addressed in future developments. We expect that this empirical study will prove valuable in advancing future models, driving LLMs and MLLMs to achieve higher accuracy at lower bit to enhance practicality. Our project is released on https://github.com/Macaronlin/LLaMA3-Quantization , and quantized models are released at https://huggingface.co/Efficient-ML .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07139v1">FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Deploying LLMs on edge devices presents serious technical challenges. Memory elasticity is crucial for edge devices with unified memory, where memory is shared and fluctuates dynamically. Existing solutions suffer from either poor transition granularity or high storage costs. We propose FlexQuant, a novel elasticity framework that generates an ensemble of quantized models, providing an elastic hosting solution with 15x granularity improvement and 10x storage reduction compared to SoTA methods. FlexQuant works with most quantization methods and creates a family of trade-off options under various storage limits through our pruning method. It brings great performance and flexibility to the edge deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06964v1">Enhancing Patient-Centric Communication: Leveraging LLMs to Simulate Patient Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing scenarios, particularly in simulating domain-specific experts using tailored prompts. This ability enables LLMs to adopt the persona of individuals with specific backgrounds, offering a cost-effective and efficient alternative to traditional, resource-intensive user studies. By mimicking human behavior, LLMs can anticipate responses based on concrete demographic or professional profiles. In this paper, we evaluate the effectiveness of LLMs in simulating individuals with diverse backgrounds and analyze the consistency of these simulated behaviors compared to real-world outcomes. In particular, we explore the potential of LLMs to interpret and respond to discharge summaries provided to patients leaving the Intensive Care Unit (ICU). We evaluate and compare with human responses the comprehensibility of discharge summaries among individuals with varying educational backgrounds, using this analysis to assess the strengths and limitations of LLM-driven simulations. Notably, when LLMs are primed with educational background information, they deliver accurate and actionable medical guidance 88% of the time. However, when other information is provided, performance significantly drops, falling below random chance levels. This preliminary study shows the potential benefits and pitfalls of automatically generating patient-specific health information from diverse populations. While LLMs show promise in simulating health personas, our results highlight critical gaps that must be addressed before they can be reliably used in clinical settings. Our findings suggest that a straightforward query-response model could outperform a more tailored approach in delivering health information. This is a crucial first step in understanding how LLMs can be optimized for personalized health communication while maintaining accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09172v4">Unlocking the Power of LLM Uncertainty for Active In-Context Example Selection</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across a wide range of downstream tasks. However, it is challenging for users to discern whether the responses of LLM are generated with certainty or are fabricated to meet user expectations. In this paper, we introduce Uncertainty Tripartite Testing Paradigm (Unc-TTP), a novel method for classifying LLM uncertainty by leveraging output inconsistency. Specifically, Unc-TTP performs three rounds of sampling under varying label injection interference, enumerating all possible outcomes, and uses the degree of output inconsistency as the indicator of the LLM's intrinsic uncertainty. To validate the effectiveness of this inconsistency-defined uncertainty, we draw inspiration from Active Learning, comparing the informativeness of actively selected in-context examples. Our experiments show that uncertainty examples selected via Unc-TTP are more informative than certainty examples. Furthermore, the Unc-TTP-guided uncertainty-based active example selection strategy outperforms existing methods, highlighting its effectiveness in classifying LLM uncertainty and enhancing in-context learning. This work not only underscores the potential of inconsistency-based uncertainty classification for both open- and closed-source LLMs but also presents a practical approach for leveraging uncertainty to improve LLM performance in real-world tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02960v2">ObfuscaTune: Obfuscated Offsite Fine-tuning and Inference of Proprietary LLMs on Private Datasets</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted at AAAI 2025 (PPAI Workshop)
    </div>
    <details class="paper-abstract">
      This work addresses the timely yet underexplored problem of performing inference and finetuning of a proprietary LLM owned by a model provider entity on the confidential/private data of another data owner entity, in a way that ensures the confidentiality of both the model and the data. Hereby, the finetuning is conducted offsite, i.e., on the computation infrastructure of a third-party cloud provider. We tackle this problem by proposing ObfuscaTune, a novel, efficient and fully utility-preserving approach that combines a simple yet effective obfuscation technique with an efficient usage of confidential computing (only 5% of the model parameters are placed on TEE). We empirically demonstrate the effectiveness of ObfuscaTune by validating it on GPT-2 models with different sizes on four NLP benchmark datasets. Finally, we compare to a na\"ive version of our approach to highlight the necessity of using random matrices with low condition numbers in our approach to reduce errors induced by the obfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06842v1">SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks, yet their training remains highly resource-intensive and susceptible to critical challenges such as training instability. A predominant source of this instability stems from gradient and loss spikes, which disrupt the learning process, often leading to costly interventions like checkpoint recovery and experiment restarts, further amplifying inefficiencies. This paper presents a comprehensive investigation into gradient spikes observed during LLM training, revealing their prevalence across multiple architectures and datasets. Our analysis shows that these spikes can be up to $1000\times$ larger than typical gradients, substantially deteriorating model performance. To address this issue, we propose Spike-Aware Adam with Momentum Reset SPAM, a novel optimizer designed to counteract gradient spikes through momentum reset and spike-aware gradient clipping. Extensive experiments, including both pre-training and fine-tuning, demonstrate that SPAM consistently surpasses Adam and its variants across various tasks, including (1) LLM pre-training from 60M to 1B, (2) 4-bit LLM pre-training,(3) reinforcement learning, and (4) Time Series Forecasting. Additionally, SPAM facilitates memory-efficient training by enabling sparse momentum, where only a subset of momentum terms are maintained and updated. When operating under memory constraints, SPAM outperforms state-of-the-art memory-efficient optimizers such as GaLore and Adam-Mini. Our work underscores the importance of mitigating gradient spikes in LLM training and introduces an effective optimization strategy that enhances both training stability and resource efficiency at scale. Code is available at https://github.com/TianjinYellow/SPAM-Optimizer.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06834v1">LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Despite its importance, studying economic behavior across diverse, non-WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations presents significant challenges. We address this issue by introducing a novel methodology that uses Large Language Models (LLMs) to create synthetic cultural agents (SCAs) representing these populations. We subject these SCAs to classic behavioral experiments, including the dictator and ultimatum games. Our results demonstrate substantial cross-cultural variability in experimental behavior. Notably, for populations with available data, SCAs' behaviors qualitatively resemble those of real human subjects. For unstudied populations, our method can generate novel, testable hypotheses about economic behavior. By integrating AI into experimental economics, this approach offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations. Our study provides a new tool for cross-cultural economic studies and demonstrates how LLMs can help experimental behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06827v1">Leveraging Taxonomy and LLMs for Improved Multimodal Hierarchical Classification</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 11 pages, 7 figures, 2 tables, and accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Multi-level Hierarchical Classification (MLHC) tackles the challenge of categorizing items within a complex, multi-layered class structure. However, traditional MLHC classifiers often rely on a backbone model with independent output layers, which tend to ignore the hierarchical relationships between classes. This oversight can lead to inconsistent predictions that violate the underlying taxonomy. Leveraging Large Language Models (LLMs), we propose a novel taxonomy-embedded transitional LLM-agnostic framework for multimodality classification. The cornerstone of this advancement is the ability of models to enforce consistency across hierarchical levels. Our evaluations on the MEP-3M dataset - a multi-modal e-commerce product dataset with various hierarchical levels - demonstrated a significant performance improvement compared to conventional LLM structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01174v2">Leveraging LLM and Text-Queried Separation for Noise-Robust Sound Event Detection</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted by ICASSP 2025 Workshop
    </div>
    <details class="paper-abstract">
      Sound Event Detection (SED) is challenging in noisy environments where overlapping sounds obscure target events. Language-queried audio source separation (LASS) aims to isolate the target sound events from a noisy clip. However, this approach can fail when the exact target sound is unknown, particularly in noisy test sets, leading to reduced performance. To address this issue, we leverage the capabilities of large language models (LLMs) to analyze and summarize acoustic data. By using LLMs to identify and select specific noise types, we implement a noise augmentation method for noise-robust fine-tuning. The fine-tuned model is applied to predict clip-wise event predictions as text queries for the LASS model. Our studies demonstrate that the proposed method improves SED performance in noisy environments. This work represents an early application of LLMs in noise-robust SED and suggests a promising direction for handling overlapping events in SED. Codes and pretrained models are available at https://github.com/apple-yinhan/Noise-robust-SED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06488v2">Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated superior multi-task capabilities, understanding the learning mechanisms behind this is still a challenging problem. In this paper, we attempt to understand such mechanisms from the perspective of neurons. Specifically, we detect task-sensitive neurons in LLMs via gradient attribution on task-specific data. Through extensive deactivation and fine-tuning experiments, we demonstrate that the detected neurons are highly correlated with the given task, which we term as task-specific neurons. With these identified task-specific neurons, we delve into two common problems in multi-task learning and continuous learning: Generalization and Catastrophic Forgetting. We find that the overlap of task-specific neurons is strongly associated with generalization and specialization across tasks. Interestingly, at certain layers of LLMs, there is a high similarity in the parameters of different task-specific neurons, and such similarity is highly correlated with the generalization performance. Inspired by these findings, we propose a neuron-level continuous fine-tuning method that only fine-tunes the current task-specific neurons during continuous learning, and extensive experiments demonstrate the effectiveness of the proposed method. Our study provides insights into the interpretability of LLMs in multi-task learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06741v1">Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of large language models (LLMs) for medical applications, ensuring the reliability and accuracy of these models in clinical settings is paramount. Existing benchmarks often focus on fixed-format tasks like multiple-choice QA, which fail to capture the complexity of real-world clinical diagnostics. Moreover, traditional evaluation metrics and LLM-based evaluators struggle with misalignment, often providing oversimplified assessments that do not adequately reflect human judgment. To address these challenges, we introduce HDCEval, a Hierarchical Divide-and-Conquer Evaluation framework tailored for fine-grained alignment in medical evaluation. HDCEval is built on a set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors, encompassing Patient Question Relevance, Medical Knowledge Correctness, and Expression. The framework decomposes complex evaluation tasks into specialized subtasks, each evaluated by expert models trained through Attribute-Driven Token Optimization (ADTO) on a meticulously curated preference dataset. This hierarchical approach ensures that each aspect of the evaluation is handled with expert precision, leading to a significant improvement in alignment with human evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17596v2">LiveIdeaBench: Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Updated author list, Corrected some issues and ref
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated remarkable capabilities in scientific tasks, existing evaluation frameworks primarily assess their performance using rich contextual inputs, overlooking their ability to generate novel ideas from minimal information. We introduce LiveIdeaBench, a comprehensive benchmark that evaluates LLMs' scientific creativity and divergent thinking capabilities using single-keyword prompts. Drawing from Guilford's creativity theory, our framework employs a dynamic panel of state-of-the-art LLMs to assess generated ideas across four key dimensions: originality, feasibility, fluency, and flexibility. Through extensive experimentation with 20 leading models across 1,180 keywords spanning 18 scientific domains, we reveal that scientific creative ability shows distinct patterns from general intelligence metrics. Notably, our results demonstrate that models like QwQ-32B-preview achieve comparable creative performance to top-tier models like o1-preview, despite significant gaps in their general intelligence scores. These findings highlight the importance of specialized evaluation frameworks for scientific creativity and suggest that the development of creative capabilities in LLMs may follow different trajectories than traditional problem-solving abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06695v1">DVM: Towards Controllable LLM Agents in Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced the capability of game agents in social deduction games (SDGs). These games rely heavily on conversation-driven interactions and require agents to infer, make decisions, and express based on such information. While this progress leads to more sophisticated and strategic non-player characters (NPCs) in SDGs, there exists a need to control the proficiency of these agents. This control not only ensures that NPCs can adapt to varying difficulty levels during gameplay, but also provides insights into the safety and fairness of LLM agents. In this paper, we present DVM, a novel framework for developing controllable LLM agents for SDGs, and demonstrate its implementation on one of the most popular SDGs, Werewolf. DVM comprises three main components: Predictor, Decider, and Discussor. By integrating reinforcement learning with a win rate-constrained decision chain reward mechanism, we enable agents to dynamically adjust their gameplay proficiency to achieve specified win rates. Experiments show that DVM not only outperforms existing methods in the Werewolf game, but also successfully modulates its performance levels to meet predefined win rate targets. These results pave the way for LLM agents' adaptive and balanced gameplay in SDGs, opening new avenues for research in controllable game agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06658v1">Comparing Few-Shot Prompting of GPT-4 LLMs with BERT Classifiers for Open-Response Assessment in Tutor Equity Training</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 8 Page Workshop Paper, AAAI2025 Workshop on Innovation and Responsibility in AI-Supported Education (iRAISE) - Open-response Grading, Feedback, Equity Training, LLMs, BERT, GPT-4
    </div>
    <details class="paper-abstract">
      Assessing learners in ill-defined domains, such as scenario-based human tutoring training, is an area of limited research. Equity training requires a nuanced understanding of context, but do contemporary large language models (LLMs) have a knowledge base that can navigate these nuances? Legacy transformer models like BERT, in contrast, have less real-world knowledge but can be more easily fine-tuned than commercial LLMs. Here, we study whether fine-tuning BERT on human annotations outperforms state-of-the-art LLMs (GPT-4o and GPT-4-Turbo) with few-shot prompting and instruction. We evaluate performance on four prediction tasks involving generating and explaining open-ended responses in advocacy-focused training lessons in a higher education student population learning to become middle school tutors. Leveraging a dataset of 243 human-annotated open responses from tutor training lessons, we find that BERT demonstrates superior performance using an offline fine-tuning approach, which is more resource-efficient than commercial GPT models. We conclude that contemporary GPT models may not adequately capture nuanced response patterns, especially in complex tasks requiring explanation. This work advances the understanding of AI-driven learner evaluation under the lens of fine-tuning versus few-shot prompting on the nuanced task of equity training, contributing to more effective training solutions and assisting practitioners in choosing adequate assessment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06628v1">Quantifying Relational Exploration in Cultural Heritage Knowledge Graphs with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      This paper introduces a neuro-symbolic approach for relational exploration in cultural heritage knowledge graphs, leveraging Large Language Models (LLMs) for explanation generation and a novel mathematical framework to quantify the interestingness of relationships. We demonstrate the importance of interestingness measure using a quantitative analysis, by highlighting its impact on the overall performance of our proposed system, particularly in terms of precision, recall, and F1-score. Using the Wikidata Cultural Heritage Linked Open Data (WCH-LOD) dataset, our approach yields a precision of 0.70, recall of 0.68, and an F1-score of 0.69, representing an improvement compared to graph-based (precision: 0.28, recall: 0.25, F1-score: 0.26) and knowledge-based baselines (precision: 0.45, recall: 0.42, F1-score: 0.43). Furthermore, our LLM-powered explanations exhibit better quality, reflected in BLEU (0.52), ROUGE-L (0.58), and METEOR (0.63) scores, all higher than the baseline approaches. We show a strong correlation (0.65) between interestingness measure and the quality of generated explanations, validating its effectiveness. The findings highlight the importance of LLMs and a mathematical formalization for interestingness in enhancing the effectiveness of relational exploration in cultural heritage knowledge graphs, with results that are measurable and testable. We further show that the system enables more effective exploration compared to purely knowledge-based and graph-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06625v1">Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 4 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in code generation tasks, yet they face significant limitations in handling complex, long-context programming challenges and demonstrating complex compositional reasoning abilities. This paper introduces a novel agentic framework for ``guided code generation'' that tries to address these limitations through a deliberately structured, fine-grained approach to code generation tasks. Our framework leverages LLMs' strengths as fuzzy searchers and approximate information retrievers while mitigating their weaknesses in long sequential reasoning and long-context understanding. Empirical evaluation using OpenAI's HumanEval benchmark with Meta's Llama 3.1 8B model (int4 precision) demonstrates a 23.79\% improvement in solution accuracy compared to direct one-shot generation. Our results indicate that structured, guided approaches to code generation can significantly enhance the practical utility of LLMs in software development while overcoming their inherent limitations in compositional reasoning and context handling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04315v2">RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    </div>
    <details class="paper-abstract">
      Fine-tuning helps large language models (LLM) recover degraded information and enhance task performance. Although Low-Rank Adaptation (LoRA) is widely used and effective for fine-tuning, we have observed that its scaling factor can limit or even reduce performance as the rank size increases. To address this issue, we propose RoRA (Rank-adaptive Reliability Optimization), a simple yet effective method for optimizing LoRA's scaling factor. By replacing $\alpha/r$ with $\alpha/\sqrt{r}$, RoRA ensures improved performance as rank size increases. Moreover, RoRA enhances low-rank adaptation in fine-tuning uncompressed models and excels in the more challenging task of accuracy recovery when fine-tuning pruned models. Extensive experiments demonstrate the effectiveness of RoRA in fine-tuning both uncompressed and pruned models. RoRA surpasses the state-of-the-art (SOTA) in average accuracy and robustness on LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B, specifically outperforming LoRA and DoRA by 6.5% and 2.9% on LLaMA-7B, respectively. In pruned model fine-tuning, RoRA shows significant advantages; for SHEARED-LLAMA-1.3, a LLaMA-7B with 81.4% pruning, RoRA achieves 5.7% higher average accuracy than LoRA and 3.9% higher than DoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06597v1">EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 7 pages, 10 figures, 5 tables. This paper has been accepted and presented at the 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC)
    </div>
    <details class="paper-abstract">
      The widespread adoption of generative AI has generated diverse opinions, with individuals expressing both support and criticism of its applications. This study investigates the emotional dynamics surrounding generative AI by analyzing human tweets referencing terms such as ChatGPT, OpenAI, Copilot, and LLMs. To further understand the emotional intelligence of ChatGPT, we examine its responses to selected tweets, highlighting differences in sentiment between human comments and LLM-generated responses. We introduce EmoXpt, a sentiment analysis framework designed to assess both human perspectives on generative AI and the sentiment embedded in ChatGPT's responses. Unlike prior studies that focus exclusively on human sentiment, EmoXpt uniquely evaluates the emotional expression of ChatGPT. Experimental results demonstrate that LLM-generated responses are notably more efficient, cohesive, and consistently positive than human responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10950v2">Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Understanding the mechanisms behind Large Language Models (LLMs) is crucial for designing improved models and strategies. While recent studies have yielded valuable insights into the mechanisms of textual LLMs, the mechanisms of Multi-modal Large Language Models (MLLMs) remain underexplored. In this paper, we apply mechanistic interpretability methods to analyze the visual question answering (VQA) mechanisms in the first MLLM, Llava. We compare the mechanisms between VQA and textual QA (TQA) in color answering tasks and find that: a) VQA exhibits a mechanism similar to the in-context learning mechanism observed in TQA; b) the visual features exhibit significant interpretability when projecting the visual embeddings into the embedding space; and c) Llava enhances the existing capabilities of the corresponding textual LLM Vicuna during visual instruction tuning. Based on these findings, we develop an interpretability tool to help users and researchers identify important visual locations for final predictions, aiding in the understanding of visual hallucination. Our method demonstrates faster and more effective results compared to existing interpretability approaches. Code: \url{https://github.com/zepingyu0512/llava-mechanism}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01339v2">Improving Sequential Recommendations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 35 pages, 12 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The sequential recommendation problem has attracted considerable research attention in the past few years, leading to the rise of numerous recommendation models. In this work, we explore how Large Language Models (LLMs), which are nowadays introducing disruptive effects in many AI-based applications, can be used to build or improve sequential recommendation approaches. Specifically, we design three orthogonal approaches and hybrids of those to leverage the power of LLMs in different ways. In addition, we investigate the potential of each approach by focusing on its comprising technical aspects and determining an array of alternative choices for each one. We conduct extensive experiments on three datasets and explore a large variety of configurations, including different language models and baseline recommendation models, to obtain a comprehensive picture of the performance of each approach. Among other observations, we highlight that initializing state-of-the-art sequential recommendation models such as BERT4Rec or SASRec with embeddings obtained from an LLM can lead to substantial performance gains in terms of accuracy. Furthermore, we find that fine-tuning an LLM for recommendation tasks enables it to learn not only the tasks, but also concepts of a domain to some extent. We also show that fine-tuning OpenAI GPT leads to considerably better performance than fine-tuning Google PaLM 2. Overall, our extensive experiments indicate a huge potential value of leveraging LLMs in future recommendation approaches. We publicly share the code and data of our experiments to ensure reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16527v2">Profiling Bias in LLMs: Stereotype Dimensions in Contextual Word Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 Accepted to NoDaLiDa/Baltic-HLT 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are the foundation of the current successes of artificial intelligence (AI), however, they are unavoidably biased. To effectively communicate the risks and encourage mitigation efforts these models need adequate and intuitive descriptions of their discriminatory properties, appropriate for all audiences of AI. We suggest bias profiles with respect to stereotype dimensions based on dictionaries from social psychology research. Along these dimensions we investigate gender bias in contextual embeddings, across contexts and layers, and generate stereotype profiles for twelve different LLMs, demonstrating their intuition and use case for exposing and visualizing bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06471v1">The Internet of Large Language Models: An Orchestration Framework for LLM Training and Knowledge Exchange Toward Artificial General Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      This paper explores the multi-dimensional challenges faced during the development of Large Language Models (LLMs), including the massive scale of model parameters and file sizes, the complexity of development environment configuration, the singularity of model functionality, and the high costs of computational resources. To address these challenges, this paper proposes three core technical solutions: LLM sharing protocol, LLM universal environment framework, and Agent optimal path module. To solve the computational resource constraints in the early stages of research, we further innovatively propose a joint mining mechanism, achieving bilateral value sharing between computing power providers and model designers, including breakthrough rewards for optimal model paths and long-term profit distribution, thereby providing researchers with cost-optimized computational resource support and promoting the continuous development of LLM research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12236v2">Enhancing LLM Agents for Code Generation with Possibility and Pass-rate Prioritized Experience Replay</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      Nowadays transformer-based Large Language Models (LLM) for code generation tasks usually apply sampling and filtering pipelines. Due to the sparse reward problem in code generation tasks caused by one-token incorrectness, transformer-based models will sample redundant programs till they find a correct one, leading to low efficiency. To overcome the challenge, we incorporate Experience Replay (ER) in the fine-tuning phase, where codes and programs produced are stored and will be replayed to give the LLM agent a chance to learn from past experiences. Based on the spirit of ER, we introduce a novel approach called BTP pipeline which consists of three phases: beam search sampling, testing phase, and prioritized experience replay phase. The approach makes use of failed programs collected by code models and replays programs with high Possibility and Pass-rate Prioritized value (P2Value) from the replay buffer to improve efficiency. P2Value comprehensively considers the possibility of transformers' output and pass rate and can make use of the redundant resources caused by the problem that most programs collected by LLMs fail to pass any tests. We empirically apply our approach in several LLMs, demonstrating that it enhances their performance in code generation tasks and surpasses existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05965v1">Model Inversion in Split Learning for Personalized LLMs: New Insights from Information Bottleneck Theory</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) have become increasingly prevalent, showcasing the impressive capabilities of models like GPT-4. This trend has also catalyzed extensive research on deploying LLMs on mobile devices. Feasible approaches for such edge-cloud deployment include using split learning. However, previous research has largely overlooked the privacy leakage associated with intermediate representations transmitted from devices to servers. This work is the first to identify model inversion attacks in the split learning framework for LLMs, emphasizing the necessity of secure defense. For the first time, we introduce mutual information entropy to understand the information propagation of Transformer-based LLMs and assess privacy attack performance for LLM blocks. To address the issue of representations being sparser and containing less information than embeddings, we propose a two-stage attack system in which the first part projects representations into the embedding space, and the second part uses a generative model to recover text from these embeddings. This design breaks down the complexity and achieves attack scores of 38%-75% in various scenarios, with an over 60% improvement over the SOTA. This work comprehensively highlights the potential privacy risks during the deployment of personalized LLMs on the edge side.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11629v5">Can Many-Shot In-Context Learning Help LLMs as Evaluators? A Preliminary Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 Accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Utilizing Large Language Models (LLMs) as evaluators to assess the performance of LLMs has garnered attention. However, this kind of evaluation approach is affected by potential biases within LLMs, raising concerns about the accuracy and reliability of the evaluation results of LLMs. To address this problem, we propose and study two many-shot In-Context Learning (ICL) prompt templates to help LLM evaluators mitigate potential biases: Many-Shot with Reference (MSwR) and Many-Shot without Reference (MSoR). Specifically, the former utilizes in-context examples with model-generated evaluation rationales as references, while the latter does not include these references. Using these prompt designs, we investigate the impact of increasing the number of in-context examples on the consistency and quality of the evaluation results. Experimental results show that advanced LLMs, such as GPT-4o, perform better in the many-shot regime than in the zero-shot and few-shot regimes. Furthermore, when using GPT-4o as an evaluator in the many-shot regime, adopting MSwR as the prompt template performs better than MSoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05926v1">LLMs Reproduce Stereotypes of Sexual and Gender Minorities</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 10 pages, 8 figures, 6 tables
    </div>
    <details class="paper-abstract">
      A large body of research has found substantial gender bias in NLP systems. Most of this research takes a binary, essentialist view of gender: limiting its variation to the categories _men_ and _women_, conflating gender with sex, and ignoring different sexual identities. But gender and sexuality exist on a spectrum, so in this paper we study the biases of large language models (LLMs) towards sexual and gender minorities beyond binary categories. Grounding our study in a widely used psychological framework -- the Stereotype Content Model -- we demonstrate that English-language survey questions about social perceptions elicit more negative stereotypes of sexual and gender minorities from LLMs, just as they do from humans. We then extend this framework to a more realistic use case: text generation. Our analysis shows that LLMs generate stereotyped representations of sexual and gender minorities in this setting, raising concerns about their capacity to amplify representational harms in creative writing, a widely promoted use case.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05891v1">Affordably Fine-tuned LLMs Provide Better Answers to Course-specific MCQs</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 The 40th ACM/SIGAPP Symposium On Applied Computing
    </div>
    <details class="paper-abstract">
      In education, the capability of generating human-like text of Large Language Models (LLMs) inspired work on how they can increase the efficiency of learning and teaching. We study the affordability of these models for educators and students by investigating how LLMs answer multiple-choice questions (MCQs) with respect to hardware constraints and refinement techniques. We explore this space by using generic pre-trained LLMs (the 7B, 13B, and 70B variants of LLaMA-2) to answer 162 undergraduate-level MCQs from a course on Programming Languages (PL) -- the MCQ dataset is a contribution of this work, which we make publicly available. Specifically, we dissect how different factors, such as using readily-available material -- (parts of) the course's textbook -- for fine-tuning and quantisation (to decrease resource usage) can change the accuracy of the responses. The main takeaway is that smaller textbook-based fine-tuned models outperform generic larger ones (whose pre-training requires conspicuous resources), making the usage of LLMs for answering MCQs resource- and material-wise affordable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05724v1">I Can't Share Code, but I need Translation -- An Empirical Study on Code Translation through Federated LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Owing to the rapid evolution of technologies and project requirements, organizations need to upgrade the code base in their software projects to a new version of the programming language or even translating to an entirely new one. However, code translation is resource-intensive and requires expertise in both the source and target languages. While researchers have made progress in automating translations between legacy and modern languages, recent work has increasingly turned to pre-trained Large Language Models (LLMs) to translate efficiently. Given the proprietary nature of code, organizations prefer fine-tuning LLMs locally rather than relying on external APIs. This is one of the first empirical studies that proposes a Federated LLM-based approach for code translation. The proposed approach enables clients to jointly train a code translator without sharing sensitive data. This study demonstrates that participants can collaboratively develop a FedLLM for efficient code translation (particularly C\# to Java and vice-versa) with superior results (more than 40\% improvement in CodeLLaMA's CodeBLEU score) compared to individual client models. Our findings indicate that FedLLM offers a collaborative approach to code translation and could serve as a promising direction for future research in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05706v1">Debugging Without Error Messages: How LLM Prompting Strategy Affects Programming Error Explanation Effectiveness</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Making errors is part of the programming process -- even for the most seasoned professionals. Novices in particular are bound to make many errors while learning. It is well known that traditional (compiler/interpreter) programming error messages have been less than helpful for many novices and can have effects such as being frustrating, containing confusing jargon, and being downright misleading. Recent work has found that large language models (LLMs) can generate excellent error explanations, but that the effectiveness of these error messages heavily depends on whether the LLM has been provided with context -- typically the original source code where the problem occurred. Knowing that programming error messages can be misleading and/or contain that serves little-to-no use (particularly for novices) we explore the reverse: what happens when GPT-3.5 is prompted for error explanations on just the erroneous source code itself -- original compiler/interpreter produced error message excluded. We utilized various strategies to make more effective error explanations, including one-shot prompting and fine-tuning. We report the baseline results of how effective the error explanations are at providing feedback, as well as how various prompting strategies might improve the explanations' effectiveness. Our results can help educators by understanding how LLMs respond to such prompts that novices are bound to make, and hopefully lead to more effective use of Generative AI in the classroom.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06386v1">Using Pre-trained LLMs for Multivariate Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) encapsulate large amounts of knowledge and take enormous amounts of compute to train. We make use of this resource, together with the observation that LLMs are able to transfer knowledge and performance from one domain or even modality to another seemingly-unrelated area, to help with multivariate demand time series forecasting. Attention in transformer-based methods requires something worth attending to -- more than just samples of a time-series. We explore different methods to map multivariate input time series into the LLM token embedding space. In particular, our novel multivariate patching strategy to embed time series features into decoder-only pre-trained Transformers produces results competitive with state-of-the-art time series forecasting models. We also use recently-developed weight-based diagnostics to validate our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06370v1">Towards a Probabilistic Framework for Analyzing and Improving LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Ensuring the reliability and verifiability of large language model (LLM)-enabled systems remains a significant challenge in software engineering. We propose a probabilistic framework for systematically analyzing and improving these systems by modeling and refining distributions over clusters of semantically equivalent outputs. This framework facilitates the evaluation and iterative improvement of Transference Models -- key software components that utilize LLMs to transform inputs into outputs for downstream tasks. To illustrate its utility, we apply the framework to the autoformalization problem, where natural language documentation is transformed into formal program specifications. Our case illustrates how probabilistic analysis enables the identification of weaknesses and guides focused alignment improvements, resulting in more reliable and interpretable outputs. This principled approach offers a foundation for addressing critical challenges in the development of robust LLM-enabled systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10427v2">Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      The field of large language models (LLMs) has grown rapidly in recent years, driven by the desire for better efficiency, interpretability, and safe use. Building on the novel approach of "activation engineering," this study explores personality modification in LLMs, drawing inspiration from research like Refusal in LLMs Is Mediated by a Single Direction (arXiv:2406.11717) and Steering Llama 2 via Contrastive Activation Addition (arXiv:2312.06681). We leverage activation engineering to develop a method for identifying and adjusting activation directions related to personality traits, which may allow for dynamic LLM personality fine-tuning. This work aims to further our understanding of LLM interpretability while examining the ethical implications of such developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v3">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 Multi-Agent AI in the Real World @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. More details on TradingAgents are available at https://TradingAgents-AI.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06322v1">Multi-Agent Collaboration Mechanisms: A Survey of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      With recent advances in Large Language Models (LLMs), Agentic AI has become phenomenal in real-world applications, moving toward multiple LLM-based agents to perceive, learn, reason, and act collaboratively. These LLM-based Multi-Agent Systems (MASs) enable groups of intelligent agents to coordinate and solve complex tasks collectively at scale, transitioning from isolated models to collaboration-centric approaches. This work provides an extensive survey of the collaborative aspect of MASs and introduces an extensible framework to guide future research. Our framework characterizes collaboration mechanisms based on key dimensions: actors (agents involved), types (e.g., cooperation, competition, or coopetition), structures (e.g., peer-to-peer, centralized, or distributed), strategies (e.g., role-based or model-based), and coordination protocols. Through a review of existing methodologies, our findings serve as a foundation for demystifying and advancing LLM-based MASs toward more intelligent and collaborative solutions for complex, real-world use cases. In addition, various applications of MASs across diverse domains, including 5G/6G networks, Industry 5.0, question answering, and social and cultural settings, are also investigated, demonstrating their wider adoption and broader impacts. Finally, we identify key lessons learned, open challenges, and potential research directions of MASs towards artificial collective intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09426v2">FlatQuant: Flatness Matters for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Recently, quantization has been widely used for the compression and acceleration of large language models~(LLMs). Due to the outliers in LLMs, it is crucial to flatten weights and activations to minimize quantization error with the equally spaced quantization points. Prior research explores various pre-quantization transformations to suppress outliers, such as per-channel scaling and Hadamard transformation. However, we observe that these transformed weights and activations can still remain steep and outspread. In this paper, we propose FlatQuant (Fast and Learnable Affine Transformation), a new post-training quantization approach to enhance flatness of weights and activations. Our approach identifies optimal affine transformations tailored to each linear layer, calibrated in hours via a lightweight objective. To reduce runtime overhead, we apply Kronecker decomposition to the transformation matrices, and fuse all operations in FlatQuant into a single kernel. Extensive experiments show that FlatQuant sets up a new state-of-the-art quantization benchmark. For instance, it achieves less than $\textbf{1}\%$ accuracy drop for W4A4 quantization on the LLaMA-3-70B model, surpassing SpinQuant by $\textbf{7.5}\%$. For inference latency, FlatQuant reduces the slowdown induced by pre-quantization transformation from 0.26x of QuaRot to merely $\textbf{0.07x}$, bringing up to $\textbf{2.3x}$ speedup for prefill and $\textbf{1.7x}$ speedup for decoding, respectively. Code is available at: \url{https://github.com/ruikangliu/FlatQuant}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06186v1">LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 15 pages, 5 Figures
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental capability for solving complex multi-step problems, particularly in visual contexts where sequential step-wise understanding is essential. Existing approaches lack a comprehensive framework for evaluating visual reasoning and do not emphasize step-wise problem-solving. To this end, we propose a comprehensive framework for advancing step-by-step visual reasoning in large language models (LMMs) through three key contributions. First, we introduce a visual reasoning benchmark specifically designed to evaluate multi-step reasoning tasks. The benchmark presents a diverse set of challenges with eight different categories ranging from complex visual perception to scientific reasoning with over 4k reasoning steps in total, enabling robust evaluation of LLMs' abilities to perform accurate and interpretable visual reasoning across multiple steps. Second, we propose a novel metric that assesses visual reasoning quality at the granularity of individual steps, emphasizing both correctness and logical coherence. The proposed metric offers deeper insights into reasoning performance compared to traditional end-task accuracy metrics. Third, we present a new multimodal visual reasoning model, named LlamaV-o1, trained using a multi-step curriculum learning approach, where tasks are progressively organized to facilitate incremental skill acquisition and problem-solving. The proposed LlamaV-o1 is designed for multi-step reasoning and learns step-by-step through a structured training paradigm. Extensive experiments show that our LlamaV-o1 outperforms existing open-source models and performs favorably against close-source proprietary models. Compared to the recent Llava-CoT, our LlamaV-o1 achieves an average score of 67.3 with an absolute gain of 3.8\% across six benchmarks while being 5 times faster during inference scaling. Our benchmark, model, and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19876v3">LUMIA: Linear probing for Unimodal and MultiModal Membership Inference Attacks leveraging internal LLM states</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in a variety of applications, but concerns around membership inference have grown in parallel. Previous efforts focus on black-to-grey-box models, thus neglecting the potential benefit from internal LLM information. To address this, we propose the use of Linear Probes (LPs) as a method to detect Membership Inference Attacks (MIAs) by examining internal activations of LLMs. Our approach, dubbed LUMIA, applies LPs layer-by-layer to get fine-grained data on the model inner workings. We test this method across several model architectures, sizes and datasets, including unimodal and multimodal tasks. In unimodal MIA, LUMIA achieves an average gain of 15.71 % in Area Under the Curve (AUC) over previous techniques. Remarkably, LUMIA reaches AUC>60% in 65.33% of cases -- an increment of 46.80% against the state of the art. Furthermore, our approach reveals key insights, such as the model layers where MIAs are most detectable. In multimodal models, LPs indicate that visual inputs can significantly contribute to detect MIAs -- AUC>60% is reached in 85.90% of experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05985v1">Exploring LLMs for Automated Pre-Testing of Cross-Cultural Surveys</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 Accepted to ICTD 2024 (Notes)
    </div>
    <details class="paper-abstract">
      Designing culturally relevant questionnaires for ICTD research is challenging, particularly when adapting surveys for populations to non-western contexts. Prior work adapted questionnaires through expert reviews and pilot studies, which are resource-intensive and time-consuming. To address these challenges, we propose using large language models (LLMs) to automate the questionnaire pretesting process in cross-cultural settings. Our study used LLMs to adapt a U.S.-focused climate opinion survey for a South African audience. We then tested the adapted questionnaire with 116 South African participants via Prolific, asking them to provide feedback on both versions. Participants perceived the LLM-adapted questions as slightly more favorable than the traditional version. Our note opens discussions on the potential role of LLMs in adapting surveys and facilitating cross-cultural questionnaire design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05981v1">Hermit Kingdom Through the Lens of Multiple Perspectives: A Case Study of LLM Hallucination on North Korea</a></div>
    <div class="paper-meta">
      📅 2025-01-10
      | 💬 Accepted at COLING 2025
    </div>
    <details class="paper-abstract">
      Hallucination in large language models (LLMs) remains a significant challenge for their safe deployment, particularly due to its potential to spread misinformation. Most existing solutions address this challenge by focusing on aligning the models with credible sources or by improving how models communicate their confidence (or lack thereof) in their outputs. While these measures may be effective in most contexts, they may fall short in scenarios requiring more nuanced approaches, especially in situations where access to accurate data is limited or determining credible sources is challenging. In this study, we take North Korea - a country characterised by an extreme lack of reliable sources and the prevalence of sensationalist falsehoods - as a case study. We explore and evaluate how some of the best-performing multilingual LLMs and specific language-based models generate information about North Korea in three languages spoken in countries with significant geo-political interests: English (United States, United Kingdom), Korean (South Korea), and Mandarin Chinese (China). Our findings reveal significant differences, suggesting that the choice of model and language can lead to vastly different understandings of North Korea, which has important implications given the global security challenges the country poses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2307.10168v3">LLMs as Workers in Human-Computational Algorithms? Replicating Crowdsourcing Pipelines with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      LLMs have shown promise in replicating human-like behavior in crowdsourcing tasks that were previously thought to be exclusive to human abilities. However, current efforts focus mainly on simple atomic tasks. We explore whether LLMs can replicate more complex crowdsourcing pipelines. We find that modern LLMs can simulate some of crowdworkers' abilities in these ``human computation algorithms,'' but the level of success is variable and influenced by requesters' understanding of LLM capabilities, the specific skills required for sub-tasks, and the optimal interaction modality for performing these sub-tasks. We reflect on human and LLMs' different sensitivities to instructions, stress the importance of enabling human-facing safeguards for LLMs, and discuss the potential of training humans and LLMs with complementary skill sets. Crucially, we show that replicating crowdsourcing pipelines offers a valuable platform to investigate 1) the relative LLM strengths on different tasks (by cross-comparing their performances on sub-tasks) and 2) LLMs' potential in complex tasks, where they can complete part of the tasks while leaving others to humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10425v3">Active Inference for Self-Organizing Multi-LLM Systems: A Bayesian Thermodynamic Approach to Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      This paper introduces a novel approach to creating adaptive language agents by integrating active inference with large language models (LLMs). While LLMs demonstrate remarkable capabilities, their reliance on static prompts limits adaptation to new information and changing environments. We address this by implementing an active inference framework that acts as a cognitive layer above an LLM-based agent, dynamically adjusting prompts and search strategies through principled information-seeking behavior. Our framework models the environment using three state factors (prompt, search, and information states) with seven observation modalities capturing quality metrics. By framing the agent's learning through the free energy principle, we enable systematic exploration of prompt combinations and search strategies. Experimental results demonstrate the effectiveness of this approach, with the agent developing accurate models of environment dynamics evidenced by emergent structure in observation matrices. Action selection patterns reveal sophisticated exploration-exploitation behavior, transitioning from initial information-gathering to targeted prompt testing. The integration of thermodynamic principles with language model capabilities provides a principled framework for creating robust, adaptable agents, extending active inference beyond traditional low-dimensional control problems to high-dimensional, language-driven environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17428v2">NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 We open-source the model at: https://huggingface.co/nvidia/NV-Embed-v2
    </div>
    <details class="paper-abstract">
      Decoder-only large language model (LLM)-based embedding models are beginning to outperform BERT or T5-based embedding models in general-purpose text embedding tasks, including dense vector-based retrieval. In this work, we introduce the NV-Embed model, incorporating architectural designs, training procedures, and curated datasets to significantly enhance the performance of LLM as a versatile embedding model, while maintaining its simplicity and reproducibility. For model architecture, we propose a latent attention layer to obtain pooled embeddings, which consistently improves retrieval and downstream task accuracy compared to mean pooling or using the last <EOS> token embedding from LLMs. To enhance representation learning, we remove the causal attention mask of LLMs during contrastive training. For training algorithm, we introduce a two-stage contrastive instruction-tuning method. It first applies contrastive training with instructions on retrieval datasets, utilizing in-batch negatives and curated hard negative examples. At stage-2, it blends various non-retrieval into instruction tuning, which not only enhances non-retrieval task accuracy but also improves retrieval performance. For training data, we utilize the hard-negative mining, synthetic data generation and existing public available datasets to boost the performance of embedding model. By combining these techniques, our NV-Embed-v1 and NV-Embed-v2 models obtained the No.1 position on the Massive Text Embedding Benchmark (MTEB) (as of May 24, 2024 and August 30, 2024, respectively) across 56 embedding tasks, demonstrating the sustained effectiveness of the proposed methods over time. Additionally, it achieved the highest scores in the Long Doc section and the second-highest scores in the QA section of the AIR Benchmark, which covers a range of out-of-domain information retrieval topics beyond those in MTEB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16337v1">Explore Activation Sparsity in Recurrent LLMs for Energy-Efficient Neuromorphic Computing</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Accepted by AICAS 2025
    </div>
    <details class="paper-abstract">
      The recent rise of Large Language Models (LLMs) has revolutionized the deep learning field. However, the desire to deploy LLMs on edge devices introduces energy efficiency and latency challenges. Recurrent LLM (R-LLM) architectures have proven effective in mitigating the quadratic complexity of self-attention, making them a potential paradigm for computing on-edge neuromorphic processors. In this work, we propose a low-cost, training-free algorithm to sparsify R-LLMs' activations to enhance energy efficiency on neuromorphic hardware. Our approach capitalizes on the inherent structure of these models, rendering them well-suited for energy-constrained environments. Although primarily designed for R-LLMs, this method can be generalized to other LLM architectures, such as transformers, as demonstrated on the OPT model, achieving comparable sparsity and efficiency improvements. Empirical studies illustrate that our method significantly reduces computational demands while maintaining competitive accuracy across multiple zero-shot learning benchmarks. Additionally, hardware simulations with the SENECA neuromorphic processor underscore notable energy savings and latency improvements. These results pave the way for low-power, real-time neuromorphic deployment of LLMs and demonstrate the feasibility of training-free on-chip adaptation using activation sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05510v1">OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Temporal Awareness, the ability to reason dynamically based on the timestamp when a question is raised, is the key distinction between offline and online video LLMs. Unlike offline models, which rely on complete videos for static, post hoc analysis, online models process video streams incrementally and dynamically adapt their responses based on the timestamp at which the question is posed. Despite its significance, temporal awareness has not been adequately evaluated in existing benchmarks. To fill this gap, we present OVO-Bench (Online-VideO-Benchmark), a novel video benchmark that emphasizes the importance of timestamps for advanced online video understanding capability benchmarking. OVO-Bench evaluates the ability of video LLMs to reason and respond to events occurring at specific timestamps under three distinct scenarios: (1) Backward tracing: trace back to past events to answer the question. (2) Real-time understanding: understand and respond to events as they unfold at the current timestamp. (3) Forward active responding: delay the response until sufficient future information becomes available to answer the question accurately. OVO-Bench comprises 12 tasks, featuring 644 unique videos and approximately human-curated 2,800 fine-grained meta-annotations with precise timestamps. We combine automated generation pipelines with human curation. With these high-quality samples, we further developed an evaluation pipeline to systematically query video LLMs along the video timeline. Evaluations of nine Video-LLMs reveal that, despite advancements on traditional benchmarks, current models struggle with online video understanding, showing a significant gap compared to human agents. We hope OVO-Bench will drive progress in video LLMs and inspire future research in online video reasoning. Our benchmark and code can be accessed at https://github.com/JoeLeelyf/OVO-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08946v2">Authorship Attribution in the Era of LLMs: Problems, Methodologies, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Accepted to ACM SIGKDD Exploration. 12 pages for the main paper. More resources and a curated list of papers are available and regularly updated at https://llm-authorship.github.io
    </div>
    <details class="paper-abstract">
      Accurate attribution of authorship is crucial for maintaining the integrity of digital content, improving forensic investigations, and mitigating the risks of misinformation and plagiarism. Addressing the imperative need for proper authorship attribution is essential to uphold the credibility and accountability of authentic authorship. The rapid advancements of Large Language Models (LLMs) have blurred the lines between human and machine authorship, posing significant challenges for traditional methods. We presents a comprehensive literature review that examines the latest research on authorship attribution in the era of LLMs. This survey systematically explores the landscape of this field by categorizing four representative problems: (1) Human-written Text Attribution; (2) LLM-generated Text Detection; (3) LLM-generated Text Attribution; and (4) Human-LLM Co-authored Text Attribution. We also discuss the challenges related to ensuring the generalization and explainability of authorship attribution methods. Generalization requires the ability to generalize across various domains, while explainability emphasizes providing transparent and understandable insights into the decisions made by these models. By evaluating the strengths and limitations of existing methods and benchmarks, we identify key open problems and future research directions in this field. This literature review serves a roadmap for researchers and practitioners interested in understanding the state of the art in this rapidly evolving field. Additional resources and a curated list of papers are available and regularly updated at https://llm-authorship.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05396v1">FairCode: Evaluating Social Bias of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capability in code generation, drawing increasing attention to the evaluation of the quality and safety of their outputs. However, research on bias in code generation remains limited. Existing studies typically assess bias by applying malicious prompts or reapply tasks and dataset for discriminative models. Given that LLMs are often aligned with human values and that prior datasets are not fully optimized for code-related tasks, there is a pressing need for benchmarks specifically designed for evaluating code models. In this study, we introduce FairCode, a novel benchmark for evaluating bias in code generation. FairCode comprises two tasks: function implementation and test case generation, each evaluating social bias through diverse scenarios. Additionally, we propose a new metric, FairScore, to assess model performance on this benchmark. We conduct experiments on widely used LLMs and provide a comprehensive analysis of the results. The findings reveal that all tested LLMs exhibit bias. The code is available at https://github.com/YongkDu/FairCode.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11672v2">MemLLM: Finetuning LLMs to Use An Explicit Read-Write Memory</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      While current large language models (LLMs) perform well on many knowledge-related tasks, they are limited by relying on their parameters as an implicit storage mechanism. As a result, they struggle with memorizing rare events and with updating their memory as facts change over time. In addition, the uninterpretable nature of parametric memory makes it challenging to prevent hallucination. Model editing and augmenting LLMs with parameters specialized for memory are only partial solutions. In this paper, we introduce MemLLM, a novel method of enhancing LLMs by integrating a structured and explicit read-and-write memory module. MemLLM tackles the aforementioned challenges by enabling dynamic interaction with the memory and improving the LLM's capabilities in using stored knowledge. Our experiments indicate that MemLLM enhances the LLM's performance and interpretability, in language modeling in general and knowledge-intensive tasks in particular. We see MemLLM as an important step towards making LLMs more grounded and factual through memory augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10747v2">Codebook LLMs: Evaluating LLMs as Measurement Tools for Political Science Concepts</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Version 2 (v1 Presented at PolMeth 2024)
    </div>
    <details class="paper-abstract">
      Codebooks -- documents that operationalize concepts and outline annotation procedures -- are used almost universally by social scientists when coding political texts. To code these texts automatically, researchers are increasing turning to generative large language models (LLMs). However, there is limited empirical evidence on whether "off-the-shelf" LLMs faithfully follow real-world codebook operationalizations and measure complex political constructs with sufficient accuracy. To address this, we gather and curate three real-world political science codebooks -- covering protest events, political violence and manifestos -- along with their unstructured texts and human labels. We also propose a five-stage framework for codebook-LLM measurement: preparing a codebook for both humans and LLMs, testing LLMs' basic capabilities on a codebook, evaluating zero-shot measurement accuracy (i.e. off-the-shelf performance), analyzing errors, and further (parameter-efficient) supervised training of LLMs. We provide an empirical demonstration of this framework using our three codebook datasets and several pretrained 7-12 billion open-weight LLMs. We find current open-weight LLMs have limitations in following codebooks zero-shot, but that supervised instruction tuning can substantially improve performance. Rather than suggesting the "best" LLM, our contribution lies in our codebook datasets, evaluation framework, and guidance for applied researchers who wish to implement their own codebook-LLM measurement projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13426v2">Safeguarding System Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 15 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized in applications where system prompts, which guide model outputs, play a crucial role. These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a robust defense mechanism designed to safeguard system prompts. PromptKeeper tackles two core challenges: reliably detecting prompt leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon detection, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05248v1">Deriving Coding-Specific Sub-Models from LLMs using Resource-Efficient Pruning</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated their exceptional performance in various complex code generation tasks. However, their broader adoption is limited by significant computational demands and high resource requirements, particularly memory and processing power. To mitigate such requirements, model pruning techniques are used to create more compact models with significantly fewer parameters. However, current approaches do not focus on the efficient extraction of programming-language-specific sub-models. In this work, we explore the idea of efficiently deriving coding-specific sub-models through unstructured pruning (i.e., Wanda). We investigate the impact of different domain-specific calibration datasets on pruning outcomes across three distinct domains and extend our analysis to extracting four language-specific sub-models: Python, Java, C++, and JavaScript. We are the first to efficiently extract programming-language-specific sub-models using appropriate calibration datasets while maintaining acceptable accuracy w.r.t. full models. We are also the first to provide analytical evidence that domain-specific tasks activate distinct regions within LLMs, supporting the creation of specialized sub-models through unstructured pruning. We believe that this work has significant potential to enhance LLM accessibility for coding by reducing computational requirements to enable local execution on consumer-grade hardware, and supporting faster inference times critical for real-time development feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05234v1">Optimizing Estonian TV Subtitles with Semi-supervised Learning and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      This paper presents an approach for generating high-quality, same-language subtitles for Estonian TV content. We fine-tune the Whisper model on human-generated Estonian subtitles and enhance it with iterative pseudo-labeling and large language model (LLM) based post-editing. Our experiments demonstrate notable subtitle quality improvement through pseudo-labeling with an unlabeled dataset. We find that applying LLM-based editing at test time enhances subtitle accuracy, while its use during training does not yield further gains. This approach holds promise for creating subtitle quality close to human standard and could be extended to real-time applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11120v2">Latent Reward: LLM-Empowered Credit Assignment in Episodic Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) often encounters delayed and sparse feedback in real-world applications, even with only episodic rewards. Previous approaches have made some progress in reward redistribution for credit assignment but still face challenges, including training difficulties due to redundancy and ambiguous attributions stemming from overlooking the multifaceted nature of mission performance evaluation. Hopefully, Large Language Model (LLM) encompasses fruitful decision-making knowledge and provides a plausible tool for reward redistribution. Even so, deploying LLM in this case is non-trivial due to the misalignment between linguistic knowledge and the symbolic form requirement, together with inherent randomness and hallucinations in inference. To tackle these issues, we introduce LaRe, a novel LLM-empowered symbolic-based decision-making framework, to improve credit assignment. Key to LaRe is the concept of the Latent Reward, which works as a multi-dimensional performance evaluation, enabling more interpretable goal attainment from various perspectives and facilitating more effective reward redistribution. We examine that semantically generated code from LLM can bridge linguistic knowledge and symbolic latent rewards, as it is executable for symbolic objects. Meanwhile, we design latent reward self-verification to increase the stability and reliability of LLM inference. Theoretically, reward-irrelevant redundancy elimination in the latent reward benefits RL performance from more accurate reward estimation. Extensive experimental results witness that LaRe (i) achieves superior temporal credit assignment to SOTA methods, (ii) excels in allocating contributions among multiple agents, and (iii) outperforms policies trained with ground truth rewards for certain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05040v1">SWE-Fixer: Training Open-Source LLMs for Effective and Efficient GitHub Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Our code, data, and model will be released at https://github.com/InternLM/SWE-Fixer
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency across a variety of complex tasks. One significant application of LLMs is in tackling software engineering challenges, particularly in resolving real-world tasks on GitHub by fixing code based on the issues reported by the users. However, many current approaches rely on proprietary LLMs, which limits reproducibility, accessibility, and transparency. The critical components of LLMs for addressing software engineering issues and how their capabilities can be effectively enhanced remain unclear. To address these challenges, we introduce SWE-Fixer, a novel open-source LLM designed to effectively and efficiently resolve GitHub issues. SWE-Fixer comprises two essential modules: a code file retrieval module and a code editing module. The retrieval module employs BM25 along with a lightweight LLM model to achieve coarse-to-fine file retrieval. Subsequently, the code editing module utilizes the other LLM model to generate patches for the identified files. Then, to mitigate the lack of publicly available datasets, we compile an extensive dataset that includes 110K GitHub issues along with their corresponding patches, and train the two modules of SWE-Fixer separately. We assess our approach on the SWE-Bench Lite and Verified benchmarks, achieving state-of-the-art performance among open-source models with scores of 23.3% and 30.2%, respectively. These outcomes highlight the efficacy of our approach. We will make our model, dataset, and code publicly available at https://github.com/InternLM/SWE-Fixer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04985v1">SpaLLM-Guard: Pairing SMS Spam Detection Using Open-source and Commercial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      The increasing threat of SMS spam, driven by evolving adversarial techniques and concept drift, calls for more robust and adaptive detection methods. In this paper, we evaluate the potential of large language models (LLMs), both open-source and commercial, for SMS spam detection, comparing their performance across zero-shot, few-shot, fine-tuning, and chain-of-thought prompting approaches. Using a comprehensive dataset of SMS messages, we assess the spam detection capabilities of prominent LLMs such as GPT-4, DeepSeek, LLAMA-2, and Mixtral. Our findings reveal that while zero-shot learning provides convenience, it is unreliable for effective spam detection. Few-shot learning, particularly with carefully selected examples, improves detection but exhibits variability across models. Fine-tuning emerges as the most effective strategy, with Mixtral achieving 98.6% accuracy and a balanced false positive and false negative rate below 2%, meeting the criteria for robust spam detection. Furthermore, we explore the resilience of these models to adversarial attacks, finding that fine-tuning significantly enhances robustness against both perceptible and imperceptible manipulations. Lastly, we investigate the impact of concept drift and demonstrate that fine-tuned LLMs, especially when combined with few-shot learning, can mitigate its effects, maintaining high performance even on evolving spam datasets. This study highlights the importance of fine-tuning and tailored learning strategies to deploy LLMs effectively for real-world SMS spam detection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02795v2">InfiFusion: A Unified Framework for Enhanced Cross-Model Reasoning via LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance across various reasoning tasks, yet building a single model that consistently excels across all domains remains challenging. This paper addresses this problem by exploring strategies to integrate multiple domain-specialized models into an efficient pivot model.We propose two fusion strategies to combine the strengths of multiple LLMs: (1) a pairwise, multi-step fusion approach that sequentially distills each source model into the pivot model, followed by a weight merging step to integrate the distilled models into the final model. This method achieves strong performance but requires substantial training effort; and (2) a unified fusion approach that aggregates all source models' outputs simultaneously.To improve the fusion process, we introduce a novel Rate-Skewness Adaptive Fusion (RSAF) technique, which dynamically adjusts top-K ratios during parameter merging for enhanced flexibility and stability.Furthermore, we propose an uncertainty-based weighting method for the unified approach, which dynamically balances the contributions of source models and outperforms other logits/distribution ensemble methods.We achieved accuracy improvements of 9.27%, 8.80%, and 8.89% on the GSM8K, MATH, and HumanEval tasks, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04961v1">Demystifying Domain-adaptive Post-training for Financial LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Domain-adaptive post-training of large language models (LLMs) has emerged as a promising approach for specialized domains such as medicine and finance. However, significant challenges remain in identifying optimal adaptation criteria and training strategies across varying data and model configurations. To address these challenges, we introduce FINDAP, a systematic and fine-grained investigation into domain-adaptive post-training of LLMs for the finance domain. Our approach begins by identifying the core capabilities required for the target domain and designing a comprehensive evaluation suite aligned with these needs. We then analyze the effectiveness of key post-training stages, including continual pretraining, instruction tuning, and preference alignment. Building on these insights, we propose an effective training recipe centered on a novel preference data distillation method, which leverages process signals from a generative reward model. The resulting model, Llama-Fin, achieves state-of-the-art performance across a wide range of financial tasks. Our analysis also highlights how each post-training stage contributes to distinct capabilities, uncovering specific challenges and effective solutions, providing valuable insights for domain adaptation of LLMs. Project page: https://github.com/SalesforceAIResearch/FinDap
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.10168v3">LLMs as Workers in Human-Computational Algorithms? Replicating Crowdsourcing Pipelines with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      LLMs have shown promise in replicating human-like behavior in crowdsourcing tasks that were previously thought to be exclusive to human abilities. However, current efforts focus mainly on simple atomic tasks. We explore whether LLMs can replicate more complex crowdsourcing pipelines. We find that modern LLMs can simulate some of crowdworkers' abilities in these ``human computation algorithms,'' but the level of success is variable and influenced by requesters' understanding of LLM capabilities, the specific skills required for sub-tasks, and the optimal interaction modality for performing these sub-tasks. We reflect on human and LLMs' different sensitivities to instructions, stress the importance of enabling human-facing safeguards for LLMs, and discuss the potential of training humans and LLMs with complementary skill sets. Crucially, we show that replicating crowdsourcing pipelines offers a valuable platform to investigate 1) the relative LLM strengths on different tasks (by cross-comparing their performances on sub-tasks) and 2) LLMs' potential in complex tasks, where they can complete part of the tasks while leaving others to humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v3">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Corrected typos & more discussion on reasoning models 33 pages, 9 figures. arXiv admin note: text overlap with arXiv:2310.05470 by other authors
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08275v4">Harnessing the Power of LLM to Support Binary Taint Analysis</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 36 pages,16 figures
    </div>
    <details class="paper-abstract">
      This paper proposes LATTE, the first static binary taint analysis that is powered by a large language model (LLM). LATTE is superior to the state of the art (e.g., Emtaint, Arbiter, Karonte) in three aspects. First, LATTE is fully automated while prior static binary taint analyzers need rely on human expertise to manually customize taint propagation rules and vulnerability inspection rules. Second, LATTE is significantly effective in vulnerability detection, demonstrated by our comprehensive evaluations. For example, LATTE has found 37 new bugs in real-world firmware which the baselines failed to find, and 7 of them have been assigned CVE numbers. Lastly, LATTE incurs remarkably low engineering cost, making it a cost-efficient and scalable solution for security researchers and practitioners. We strongly believe that LATTE opens up a new direction to harness the recent advance in LLMs to improve vulnerability analysis for binary programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04908v1">HaVen: Hallucination-Mitigated LLM for Verilog Code Generation Aligned with HDL Engineers</a></div>
    <div class="paper-meta">
      📅 2025-01-09
    </div>
    <details class="paper-abstract">
      Recently, the use of large language models (LLMs) for Verilog code generation has attracted great research interest to enable hardware design automation. However, previous works have shown a gap between the ability of LLMs and the practical demands of hardware description language (HDL) engineering. This gap includes differences in how engineers phrase questions and hallucinations in the code generated. To address these challenges, we introduce HaVen, a novel LLM framework designed to mitigate hallucinations and align Verilog code generation with the practices of HDL engineers. HaVen tackles hallucination issues by proposing a comprehensive taxonomy and employing a chain-of-thought (CoT) mechanism to translate symbolic modalities (e.g. truth tables, state diagrams, etc.) into accurate natural language descriptions. Furthermore, HaVen bridges this gap by using a data augmentation strategy. It synthesizes high-quality instruction-code pairs that match real HDL engineering practices. Our experiments demonstrate that HaVen significantly improves the correctness of Verilog code generation, outperforming state-of-the-art LLM-based Verilog generation methods on VerilogEval and RTLLM benchmark. HaVen is publicly available at https://github.com/Intelligent-Computing-Research-Group/HaVen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04904v1">JELLY: Joint Emotion Recognition and Context Reasoning with LLMs for Conversational Speech Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-01-09
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing demand for conversational speech synthesis (CSS) that generates more natural speech by considering the conversational context. To address this, we introduce JELLY, a novel CSS framework that integrates emotion recognition and context reasoning for generating appropriate speech in conversation by fine-tuning a large language model (LLM) with multiple partial LoRA modules. We propose an Emotion-aware Q-former encoder, which enables the LLM to perceive emotions in speech. The encoder is trained to align speech emotions with text, utilizing datasets of emotional speech. The entire model is then fine-tuned with conversational speech data to infer emotional context for generating emotionally appropriate speech in conversation. Our experimental results demonstrate that JELLY excels in emotional context modeling, synthesizing speech that naturally aligns with conversation, while mitigating the scarcity of emotional conversational speech datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03489v2">Entropy-Guided Attention for Private LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Accepted to the 6th AAAI Workshop on Privacy-Preserving Artificial Intelligence (PPAI), 2025. arXiv admin note: substantial text overlap with arXiv:2410.13060
    </div>
    <details class="paper-abstract">
      The pervasiveness of proprietary language models has raised critical privacy concerns, necessitating advancements in private inference (PI), where computations are performed directly on encrypted data without revealing users' sensitive information. While PI offers a promising solution, its practical deployment is hindered by substantial communication and latency overheads, primarily stemming from nonlinear operations. To address this, we introduce an information-theoretic framework to characterize the role of nonlinearities in decoder-only language models, laying a principled foundation for optimizing transformer-architectures tailored to the demands of PI. By leveraging Shannon's entropy as a quantitative measure, we uncover the previously unexplored dual significance of nonlinearities: beyond ensuring training stability, they are crucial for maintaining attention head diversity. Specifically, we find that their removal triggers two critical failure modes: {\em entropy collapse} in deeper layers that destabilizes training, and {\em entropic overload} in earlier layers that leads to under-utilization of Multi-Head Attention's (MHA) representational capacity. We propose an entropy-guided attention mechanism paired with a novel entropy regularization technique to mitigate entropic overload. Additionally, we explore PI-friendly alternatives to layer normalization for preventing entropy collapse and stabilizing the training of LLMs with reduced-nonlinearities. Our study bridges the gap between information theory and architectural design, establishing entropy dynamics as a principled guide for developing efficient PI architectures. The code and implementation are available at https://github.com/Nandan91/entropy-guided-attention-llm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04835v1">Do Code LLMs Understand Design Patterns?</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 accpeted by llm4code workshop in ICSE 2025
    </div>
    <details class="paper-abstract">
      Code Large Language Models (LLMs) demonstrate great versatility in adapting to various downstream tasks, including code generation and completion, as well as bug detection and fixing. However, Code LLMs often fail to capture existing coding standards, leading to the generation of code that conflicts with the required design patterns for a given project. As a result, developers must post-process to adapt the generated code to the project's design norms. In this work, we empirically investigate the biases of Code LLMs in software development. Through carefully designed experiments, we assess the models' understanding of design patterns across recognition, comprehension, and generation. Our findings reveal that biases in Code LLMs significantly affect the reliability of downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04682v1">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      We propose a novel framework, Meta Chain-of-Thought (Meta-CoT), which extends traditional Chain-of-Thought (CoT) by explicitly modeling the underlying reasoning required to arrive at a particular CoT. We present empirical evidence from state-of-the-art models exhibiting behaviors consistent with in-context search, and explore methods for producing Meta-CoT via process supervision, synthetic data generation, and search algorithms. Finally, we outline a concrete pipeline for training a model to produce Meta-CoTs, incorporating instruction tuning with linearized search traces and reinforcement learning post-training. Finally, we discuss open research questions, including scaling laws, verifier roles, and the potential for discovering novel reasoning algorithms. This work provides a theoretical and practical roadmap to enable Meta-CoT in LLMs, paving the way for more powerful and human-like reasoning in artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04670v1">Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 project page: https://zhouyiks.github.io/projects/CoLVA/
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal models have shown a strong ability in visual perception, reasoning abilities, and vision-language understanding. However, studies on visual matching ability are missing, where finding the visual correspondence of objects is essential in vision research. Our research reveals that the matching capabilities in recent multimodal LLMs (MLLMs) still exhibit systematic shortcomings, even with current strong MLLMs models, GPT-4o. In particular, we construct a Multimodal Visual Matching (MMVM) benchmark to fairly benchmark over 30 different MLLMs. The MMVM benchmark is built from 15 open-source datasets and Internet videos with manual annotation. We categorize the data samples of MMVM benchmark into eight aspects based on the required cues and capabilities to more comprehensively evaluate and analyze current MLLMs. In addition, we have designed an automatic annotation pipeline to generate the MMVM SFT dataset, including 220K visual matching data with reasoning annotation. Finally, we present CoLVA, a novel contrastive MLLM with two novel technical designs: fine-grained vision expert with object-level contrastive learning and instruction augmentation strategy. CoLVA achieves 51.06\% overall accuracy (OA) on the MMVM benchmark, surpassing GPT-4o and baseline by 8.41\% and 23.58\% OA, respectively. The results show the effectiveness of our MMVM SFT dataset and our novel technical designs. Code, benchmark, dataset, and models are available at https://github.com/zhouyiks/CoLVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04648v1">FlairGPT: Repurposing LLMs for Interior Designs</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Accepted at EUROGRAPHICS 2025
    </div>
    <details class="paper-abstract">
      Interior design involves the careful selection and arrangement of objects to create an aesthetically pleasing, functional, and harmonized space that aligns with the client's design brief. This task is particularly challenging, as a successful design must not only incorporate all the necessary objects in a cohesive style, but also ensure they are arranged in a way that maximizes accessibility, while adhering to a variety of affordability and usage considerations. Data-driven solutions have been proposed, but these are typically room- or domain-specific and lack explainability in their design design considerations used in producing the final layout. In this paper, we investigate if large language models (LLMs) can be directly utilized for interior design. While we find that LLMs are not yet capable of generating complete layouts, they can be effectively leveraged in a structured manner, inspired by the workflow of interior designers. By systematically probing LLMs, we can reliably generate a list of objects along with relevant constraints that guide their placement. We translate this information into a design layout graph, which is then solved using an off-the-shelf constrained optimization setup to generate the final layouts. We benchmark our algorithm in various design configurations against existing LLM-based methods and human designs, and evaluate the results using a variety of quantitative and qualitative metrics along with user studies. In summary, we demonstrate that LLMs, when used in a structured manner, can effectively generate diverse high-quality layouts, making them a viable solution for creating large-scale virtual scenes. Project webpage at https://flairgpt.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00957v3">Generative AI and LLMs in Industry: A text-mining Analysis and Critical Evaluation of Guidelines and Policy Statements Across Fourteen Industrial Sectors</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      The rise of Generative AI (GAI) and Large Language Models (LLMs) has transformed industrial landscapes, offering unprecedented opportunities for efficiency and innovation while raising critical ethical, regulatory, and operational challenges. This study conducts a text-based analysis of 160 guidelines and policy statements across fourteen industrial sectors, utilizing systematic methods and text-mining techniques to evaluate the governance of these technologies. By examining global directives, industry practices, and sector-specific policies, the paper highlights the complexities of balancing innovation with ethical accountability and equitable access. The findings provide actionable insights and recommendations for fostering responsible, transparent, and safe integration of GAI and LLMs in diverse industry contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13375v2">Extending LLMs to New Languages: A Case Study of Llama and Persian Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 accepted at COLING 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made great progress in classification and text generation tasks. However, they are mainly trained on English data and often struggle with low-resource languages. In this study, we explore adding a new language, i.e., Persian, to Llama (a model with a limited understanding of Persian) using parameter-efficient fine-tuning. We employ a multi-stage approach involving pretraining on monolingual Persian data, aligning representations through bilingual pretraining and instruction datasets, and instruction-tuning with task-specific datasets. We evaluate the model's performance at each stage on generation and classification tasks. Our findings suggest that incorporating the Persian language, through bilingual data alignment, can enhance classification accuracy for Persian tasks, with no adverse impact and sometimes even improvements on English tasks. Additionally, the results highlight the model's initial strength as a critical factor when working with limited training data, with cross-lingual alignment offering minimal benefits for the low-resource language. Knowledge transfer from English to Persian has a marginal effect, primarily benefiting simple classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00599v2">VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 17 pages, 14 figures, technical report
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have recently exhibited remarkable capabilities in general video understanding. However, they mainly focus on holistic comprehension and struggle with capturing fine-grained spatial and temporal details. Besides, the lack of high-quality object-level video instruction data and a comprehensive benchmark further hinders their advancements. To tackle these challenges, we introduce the VideoRefer Suite to empower Video LLM for finer-level spatial-temporal video understanding, i.e., enabling perception and reasoning on any objects throughout the video. Specially, we thoroughly develop VideoRefer Suite across three essential aspects: dataset, model, and benchmark. Firstly, we introduce a multi-agent data engine to meticulously curate a large-scale, high-quality object-level video instruction dataset, termed VideoRefer-700K. Next, we present the VideoRefer model, which equips a versatile spatial-temporal object encoder to capture precise regional and sequential representations. Finally, we meticulously create a VideoRefer-Bench to comprehensively assess the spatial-temporal understanding capability of a Video LLM, evaluating it across various aspects. Extensive experiments and analyses demonstrate that our VideoRefer model not only achieves promising performance on video referring benchmarks but also facilitates general video understanding capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13111v1">iServe: An Intent-based Serving System for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 19 pages, 24 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming ubiquitous across industries, where applications demand they fulfill diverse user intents. However, developers currently face the challenge of manually exploring numerous deployment configurations - combinations of parallelism and compression techniques that impact resource usage, latency, cost, and accuracy - to meet these intents. Assessing the impact of these configurations on user metrics requires extensive, costly profiling for each model. Existing approaches avoid this expense by using fixed, static configurations, but this often leads to sub-optimal performance and higher costs. Moreover, none of these solutions dynamically adapt to changing user intents to balance latency and cost, effectively. We present iServe, an automated, intent-based system for distributed LLM inference. Instead of manually selecting deployment configurations, developers simply specify their intent - such as minimizing latency, reducing cost, or meeting specific targets for either. iServe introduces fingerprints, lightweight representations of LLMs, to efficiently estimate how different configurations impact latency and memory usage. Based on these insights and GPU availability, iServe dynamically selects the optimal configuration to align with the user's intent. For various LLMs and query arrival rates, iServe best meets user intents compared to state-of-the-art systems by reducing latency by 77.62% and SLO violations by 7.09x while improving GPU throughput by 4.72x. Moreover, iServe's fingerprint-based profiling reduces profiling cost by 6.05x (GPU-hours) compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04473v1">When LLMs Struggle: Reference-less Translation Evaluation for Low-resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      This paper investigates the reference-less evaluation of machine translation for low-resource language pairs, known as quality estimation (QE). Segment-level QE is a challenging cross-lingual language understanding task that provides a quality score (0-100) to the translated output. We comprehensively evaluate large language models (LLMs) in zero/few-shot scenarios and perform instruction fine-tuning using a novel prompt based on annotation guidelines. Our results indicate that prompt-based approaches are outperformed by the encoder-based fine-tuned QE models. Our error analysis reveals tokenization issues, along with errors due to transliteration and named entities, and argues for refinement in LLM pre-training for cross-lingual tasks. We release the data, and models trained publicly for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04437v1">Integrating LLMs with ITS: Recent Advances, Potentials, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Accepted for publication in IEEE Transactions on Intelligent Transportation Systems
    </div>
    <details class="paper-abstract">
      Intelligent Transportation Systems (ITS) are crucial for the development and operation of smart cities, addressing key challenges in efficiency, productivity, and environmental sustainability. This paper comprehensively reviews the transformative potential of Large Language Models (LLMs) in optimizing ITS. Initially, we provide an extensive overview of ITS, highlighting its components, operational principles, and overall effectiveness. We then delve into the theoretical background of various LLM techniques, such as GPT, T5, CTRL, and BERT, elucidating their relevance to ITS applications. Following this, we examine the wide-ranging applications of LLMs within ITS, including traffic flow prediction, vehicle detection and classification, autonomous driving, traffic sign recognition, and pedestrian detection. Our analysis reveals how these advanced models can significantly enhance traffic management and safety. Finally, we explore the challenges and limitations LLMs face in ITS, such as data availability, computational constraints, and ethical considerations. We also present several future research directions and potential innovations to address these challenges. This paper aims to guide researchers and practitioners through the complexities and opportunities of integrating LLMs in ITS, offering a roadmap to create more efficient, sustainable, and responsive next-generation transportation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04436v1">Federated Fine-Tuning of LLMs: Framework Comparison and Research Directions</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Federated learning (FL) provides a privacy-preserving solution for fine-tuning pre-trained large language models (LLMs) using distributed private datasets, enabling task-specific adaptation while preserving data privacy. However, fine-tuning the extensive parameters in LLMs is particularly challenging in resource-constrained federated scenarios due to the significant communication and computational costs. To gain a deeper understanding of how these challenges can be addressed, this article conducts a comparative analysis three advanced federated LLM (FedLLM) frameworks that integrate knowledge distillation (KD) and split learning (SL) to mitigate these issues: 1) FedLLMs, where clients upload model parameters or gradients to enable straightforward and effective fine-tuning; 2) KD-FedLLMs, which leverage KD for efficient knowledge sharing via logits; and 3) Split-FedLLMs, which split the LLMs into two parts, with one part executed on the client and the other one on the server, to balance the computational load. Each framework is evaluated based on key performance metrics, including model accuracy, communication overhead, and client-side computational load, offering insights into their effectiveness for various federated fine-tuning scenarios. Through this analysis, we identify framework-specific optimization opportunities to enhance the efficiency of FedLLMs and discuss broader research directions, highlighting open opportunities to better adapt FedLLMs for real-world applications. A use case is presented to demonstrate the performance comparison of these three frameworks under varying configurations and settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10587v3">RDRec: Rationale Distillation for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 10 pages. Accepted to ACL 2024 Main as a short paper
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based recommender models that bridge users and items through textual prompts for effective semantic reasoning have gained considerable attention. However, few methods consider the underlying rationales behind interactions, such as user preferences and item attributes, limiting the reasoning capability of LLMs for recommendations. This paper proposes a rationale distillation recommender (RDRec), a compact model designed to learn rationales generated by a larger language model (LM). By leveraging rationales from reviews related to users and items, RDRec remarkably specifies their profiles for recommendations. Experiments show that RDRec achieves state-of-the-art (SOTA) performance in both top-N and sequential recommendations. Our source code is released at https://github.com/WangXFng/RDRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03535v2">SenseRAG: Constructing Environmental Knowledge Bases with Proactive Querying for LLM-Based Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 This paper has been accepted for presentation at WACV Workshop LLMAD 2025
    </div>
    <details class="paper-abstract">
      This study addresses the critical need for enhanced situational awareness in autonomous driving (AD) by leveraging the contextual reasoning capabilities of large language models (LLMs). Unlike traditional perception systems that rely on rigid, label-based annotations, it integrates real-time, multimodal sensor data into a unified, LLMs-readable knowledge base, enabling LLMs to dynamically understand and respond to complex driving environments. To overcome the inherent latency and modality limitations of LLMs, a proactive Retrieval-Augmented Generation (RAG) is designed for AD, combined with a chain-of-thought prompting mechanism, ensuring rapid and context-rich understanding. Experimental results using real-world Vehicle-to-everything (V2X) datasets demonstrate significant improvements in perception and prediction performance, highlighting the potential of this framework to enhance safety, adaptability, and decision-making in next-generation AD systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04336v1">Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Long-form video understanding with Large Vision Language Models is challenged by the need to analyze temporally dispersed yet spatially concentrated key moments within limited context windows. In this work, we introduce VideoMindPalace, a new framework inspired by the "Mind Palace", which organizes critical video moments into a topologically structured semantic graph. VideoMindPalace organizes key information through (i) hand-object tracking and interaction, (ii) clustered activity zones representing specific areas of recurring activities, and (iii) environment layout mapping, allowing natural language parsing by LLMs to provide grounded insights on spatio-temporal and 3D context. In addition, we propose the Video MindPalace Benchmark (VMB), to assess human-like reasoning, including spatial localization, temporal reasoning, and layout-aware sequential understanding. Evaluated on VMB and established video QA datasets, including EgoSchema, NExT-QA, IntentQA, and the Active Memories Benchmark, VideoMindPalace demonstrates notable gains in spatio-temporal coherence and human-aligned reasoning, advancing long-form video analysis capabilities in VLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07464v2">BudgetMLAgent: A Cost-Effective LLM Multi-Agent system for Automating Machine Learning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Presented at AIMLSystems '24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in diverse applications including generation of code snippets, but often struggle with generating code for complex Machine Learning (ML) tasks. Although existing LLM single-agent based systems give varying performance depending on the task complexity, they purely rely on larger and expensive models such as GPT-4. Our investigation reveals that no-cost and low-cost models such as Gemini-Pro, Mixtral and CodeLlama perform far worse than GPT-4 in a single-agent setting. With the motivation of developing a cost-efficient LLM based solution for solving ML tasks, we propose an LLM Multi-Agent based system which leverages combination of experts using profiling, efficient retrieval of past observations, LLM cascades, and ask-the-expert calls. Through empirical analysis on ML engineering tasks in the MLAgentBench benchmark, we demonstrate the effectiveness of our system, using no-cost models, namely Gemini as the base LLM, paired with GPT-4 in cascade and expert to serve occasional ask-the-expert calls for planning. With 94.2\% reduction in the cost (from \$0.931 per run cost averaged over all tasks for GPT-4 single agent system to \$0.054), our system is able to yield better average success rate of 32.95\% as compared to GPT-4 single-agent system yielding 22.72\% success rate averaged over all the tasks of MLAgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14418v3">MEDSAGE: Enhancing Robustness of Medical Dialogue Summarization to ASR Errors with LLM-generated Synthetic Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Accepted by the Thirty-Ninth AAAI Conference on Artificial Intelligence (AAAI-25)
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) systems are pivotal in transcribing speech into text, yet the errors they introduce can significantly degrade the performance of downstream tasks like summarization. This issue is particularly pronounced in clinical dialogue summarization, a low-resource domain where supervised data for fine-tuning is scarce, necessitating the use of ASR models as black-box solutions. Employing conventional data augmentation for enhancing the noise robustness of summarization models is not feasible either due to the unavailability of sufficient medical dialogue audio recordings and corresponding ASR transcripts. To address this challenge, we propose MEDSAGE, an approach for generating synthetic samples for data augmentation using Large Language Models (LLMs). Specifically, we leverage the in-context learning capabilities of LLMs and instruct them to generate ASR-like errors based on a few available medical dialogue examples with audio recordings. Experimental results show that LLMs can effectively model ASR noise, and incorporating this noisy data into the training process significantly improves the robustness and accuracy of medical dialogue summarization systems. This approach addresses the challenges of noisy ASR outputs in critical applications, offering a robust solution to enhance the reliability of clinical dialogue summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11102v2">Empowering LLMs to Understand and Generate Complex Vector Graphics</a></div>
    <div class="paper-meta">
      📅 2025-01-08
      | 💬 Project Page: https://ximinng.github.io/LLM4SVGProject/
    </div>
    <details class="paper-abstract">
      The unprecedented advancements in Large Language Models (LLMs) have profoundly impacted natural language processing but have yet to fully embrace the realm of scalable vector graphics (SVG) generation. While LLMs encode partial knowledge of SVG data from web pages during training, recent findings suggest that semantically ambiguous and tokenized representations within LLMs may result in hallucinations in vector primitive predictions. Additionally, LLM training typically lacks modeling and understanding of the rendering sequence of vector paths, which can lead to occlusion between output vector primitives. In this paper, we present LLM4SVG, an initial yet substantial step toward bridging this gap by enabling LLMs to better understand and generate vector graphics. LLM4SVG facilitates a deeper understanding of SVG components through learnable semantic tokens, which precisely encode these tokens and their corresponding properties to generate semantically aligned SVG outputs. Using a series of learnable semantic tokens, a structured dataset for instruction following is developed to support comprehension and generation across two primary tasks. Our method introduces a modular architecture to existing large language models, integrating semantic tags, vector instruction encoders, fine-tuned commands, and powerful LLMs to tightly combine geometric, appearance, and language information. To overcome the scarcity of SVG-text instruction data, we developed an automated data generation pipeline that collected a massive dataset of more than 250k SVG data and 580k SVG-text instructions, which facilitated the adoption of the two-stage training strategy popular in LLM development. By exploring various training strategies, we developed LLM4SVG, which significantly moves beyond optimized rendering-based approaches and language-model-based baselines to achieve remarkable results in human evaluation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04285v1">Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Along with the proliferating research interest in Semantic Communication (SemCom), Joint Source Channel Coding (JSCC) has dominated the attention due to the widely assumed existence in efficiently delivering information semantics. %has emerged as a pivotal area of research, aiming to enhance the efficiency and reliability of information transmission through deep learning-based methods. Nevertheless, this paper challenges the conventional JSCC paradigm, and advocates for adoption of Separate Source Channel Coding (SSCC) to enjoy the underlying more degree of freedom for optimization. We demonstrate that SSCC, after leveraging the strengths of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, offers superior performance over JSCC. Our proposed framework also effectively highlights the compatibility challenges between SemCom approaches and digital communication systems, particularly concerning the resource costs associated with the transmission of high precision floating point numbers. Through comprehensive evaluations, we establish that empowered by LLM-based compression and ECCT-enhanced error correction, SSCC remains a viable and effective solution for modern communication systems. In other words, separate source and channel coding is still what we need!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04249v1">IOLBENCH: Benchmarking LLMs on Linguistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Despite the remarkable advancements and widespread applications of deep neural networks, their ability to perform reasoning tasks remains limited, particularly in domains requiring structured, abstract thought. In this paper, we investigate the linguistic reasoning capabilities of state-of-the-art large language models (LLMs) by introducing IOLBENCH, a novel benchmark derived from International Linguistics Olympiad (IOL) problems. This dataset encompasses diverse problems testing syntax, morphology, phonology, and semantics, all carefully designed to be self-contained and independent of external knowledge. These tasks challenge models to engage in metacognitive linguistic reasoning, requiring the deduction of linguistic rules and patterns from minimal examples. Through extensive benchmarking of leading LLMs, we find that even the most advanced models struggle to handle the intricacies of linguistic complexity, particularly in areas demanding compositional generalization and rule abstraction. Our analysis highlights both the strengths and persistent limitations of current models in linguistic problem-solving, offering valuable insights into their reasoning capabilities. By introducing IOLBENCH, we aim to foster further research into developing models capable of human-like reasoning, with broader implications for the fields of computational linguistics and artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13586v2">Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Sampling-based decoding strategies have been widely adopted for Large Language Models (LLMs) in numerous applications, targeting a balance between diversity and quality via temperature tuning and tail truncation. Considering the strong dependency of the candidate next tokens on different prefixes, recent studies propose to adaptively truncate the tail of LLMs' predicted distribution. Although improved results have been reported with these methods on open-ended text generation tasks, the results are highly dependent on the curated parameters and the limited exemplar text. In this paper, we propose a systematic way to estimate the capacity of a truncation sampling method by considering the trade-off between diversity and risk at each decoding step, based on our collected prefix tree which preserves the context of a full sentence. Our work offers a comprehensive comparison of existing truncation sampling methods and serves as a practical user guideline for their parameter selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04227v1">Agent Laboratory: Using LLM Agents as Research Assistants</a></div>
    <div class="paper-meta">
      📅 2025-01-08
    </div>
    <details class="paper-abstract">
      Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03747v1">Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 no comment
    </div>
    <details class="paper-abstract">
      Recently, leveraging pre-trained Large Language Models (LLMs) for time series (TS) tasks has gained increasing attention, which involves activating and enhancing LLMs' capabilities. Many methods aim to activate LLMs' capabilities based on token-level alignment but overlook LLMs' inherent strength on natural language processing -- their deep understanding of linguistic logic and structure rather than superficial embedding processing. We propose Context-Alignment, a new paradigm that aligns TS with a linguistic component in the language environments familiar to LLMs to enable LLMs to contextualize and comprehend TS data, thereby activating their capabilities. Specifically, such context-level alignment comprises structural alignment and logical alignment, which is achieved by a Dual-Scale Context-Alignment GNNs (DSCA-GNNs) applied to TS-language multimodal inputs. Structural alignment utilizes dual-scale nodes to describe hierarchical structure in TS-language, enabling LLMs treat long TS data as a whole linguistic component while preserving intrinsic token features. Logical alignment uses directed edges to guide logical relationships, ensuring coherence in the contextual semantics. Demonstration examples prompt are employed to construct Demonstration Examples based Context-Alignment (DECA) following DSCA-GNNs framework. DECA can be flexibly and repeatedly integrated into various layers of pre-trained LLMs to improve awareness of logic and structure, thereby enhancing performance. Extensive experiments show the effectiveness of DECA and the importance of Context-Alignment across tasks, particularly in few-shot and zero-shot forecasting, confirming that Context-Alignment provide powerful prior knowledge on context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12761v2">BTGenBot: Behavior Tree Generation for Robotic Tasks with Lightweight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to generating behavior trees for robots using lightweight large language models (LLMs) with a maximum of 7 billion parameters. The study demonstrates that it is possible to achieve satisfying results with compact LLMs when fine-tuned on a specific dataset. The key contributions of this research include the creation of a fine-tuning dataset based on existing behavior trees using GPT-3.5 and a comprehensive comparison of multiple LLMs (namely llama2, llama-chat, and code-llama) across nine distinct tasks. To be thorough, we evaluated the generated behavior trees using static syntactical analysis, a validation system, a simulated environment, and a real robot. Furthermore, this work opens the possibility of deploying such solutions directly on the robot, enhancing its practical applicability. Findings from this study demonstrate the potential of LLMs with a limited number of parameters in generating effective and efficient robot behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.17624v2">Forecasting Credit Ratings: A Case Study where Traditional Methods Outperform Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been shown to perform well for many downstream tasks. Transfer learning can enable LLMs to acquire skills that were not targeted during pre-training. In financial contexts, LLMs can sometimes beat well-established benchmarks. This paper investigates how well LLMs perform in the task of forecasting corporate credit ratings. We show that while LLMs are very good at encoding textual information, traditional methods are still very competitive when it comes to encoding numeric and multimodal data. For our task, current LLMs perform worse than a more traditional XGBoost architecture that combines fundamental and macroeconomic data with high-density text-based embedding features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05475v1">Retrieval-Augmented Generation by Evidence Retroactivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation has gained significant attention due to its ability to integrate relevant external knowledge, enhancing the accuracy and reliability of the LLMs' responses. Most of the existing methods apply a dynamic multiple retrieval-generating process, to address multi-hop complex questions by decomposing them into sub-problems. However, these methods rely on an unidirectional forward reasoning paradigm, where errors from insufficient reasoning steps or inherent flaws in current retrieval systems are irreversible, potentially derailing the entire reasoning chain. For the first time, this work introduces Retroactive Retrieval-Augmented Generation (RetroRAG), a novel framework to build a retroactive reasoning paradigm. RetroRAG revises and updates the evidence, redirecting the reasoning chain to the correct direction. RetroRAG constructs an evidence-collation-discovery framework to search, generate, and refine credible evidence. It synthesizes inferential evidence related to the key entities in the question from the existing source knowledge and formulates search queries to uncover additional information. As new evidence is found, RetroRAG continually updates and organizes this information, enhancing its ability to locate further necessary evidence. Paired with an Answerer to generate and evaluate outputs, RetroRAG is capable of refining its reasoning process iteratively until a reliable answer is obtained. Empirical evaluations show that RetroRAG significantly outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03569v1">What Does a Software Engineer Look Like? Exploring Societal Stereotypes in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Accepted for publication in Software Engineering in Society (SEIS) in ICSE 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have rapidly gained popularity and are being embedded into professional applications due to their capabilities in generating human-like content. However, unquestioned reliance on their outputs and recommendations can be problematic as LLMs can reinforce societal biases and stereotypes. This study investigates how LLMs, specifically OpenAI's GPT-4 and Microsoft Copilot, can reinforce gender and racial stereotypes within the software engineering (SE) profession through both textual and graphical outputs. We used each LLM to generate 300 profiles, consisting of 100 gender-based and 50 gender-neutral profiles, for a recruitment scenario in SE roles. Recommendations were generated for each profile and evaluated against the job requirements for four distinct SE positions. Each LLM was asked to select the top 5 candidates and subsequently the best candidate for each role. Each LLM was also asked to generate images for the top 5 candidates, providing a dataset for analysing potential biases in both text-based selections and visual representations. Our analysis reveals that both models preferred male and Caucasian profiles, particularly for senior roles, and favoured images featuring traits such as lighter skin tones, slimmer body types, and younger appearances. These findings highlight underlying societal biases influence the outputs of LLMs, contributing to narrow, exclusionary stereotypes that can further limit diversity and perpetuate inequities in the SE field. As LLMs are increasingly adopted within SE research and professional practices, awareness of these biases is crucial to prevent the reinforcement of discriminatory norms and to ensure that AI tools are leveraged to promote an inclusive and equitable engineering culture rather than hinder it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02155v2">CausalMob: Causal Human Mobility Prediction with LLMs-derived Human Intentions toward Public Events</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Large-scale human mobility exhibits spatial and temporal patterns that can assist policymakers in decision making. Although traditional prediction models attempt to capture these patterns, they often interfered by non-periodic public events, such as disasters and occasional celebrations. Since regular human mobility patterns are heavily affected by these events, estimating their causal effects is critical to accurate mobility predictions. Although news articles provide unique perspectives on these events in an unstructured format, processing is a challenge. In this study, we propose a causality-augmented prediction model, called CausalMob, to analyze the causal effects of public events. We first utilize large language models (LLMs) to extract human intentions from news articles and transform them into features that act as causal treatments. Next, the model learns representations of spatio-temporal regional covariates from multiple data sources to serve as confounders for causal inference. Finally, we present a causal effect estimation framework to ensure event features remain independent of confounders during prediction. Based on large-scale real-world data, the experimental results show that the proposed model excels in human mobility prediction, outperforming state-of-the-art models.
    </details>
</div>
