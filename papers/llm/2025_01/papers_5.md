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
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05482v1">HP-BERT: A framework for longitudinal study of Hinduphobia on social media via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      During the COVID-19 pandemic, community tensions intensified, fuelling Hinduphobic sentiments and discrimination against individuals of Hindu descent within India and worldwide. Large language models (LLMs) have become prominent in natural language processing (NLP) tasks and social media analysis, enabling longitudinal studies of platforms like X (formerly Twitter) for specific issues during COVID-19. We present an abuse detection and sentiment analysis framework that offers a longitudinal analysis of Hinduphobia on X (Twitter) during and after the COVID-19 pandemic. This framework assesses the prevalence and intensity of Hinduphobic discourse, capturing elements such as derogatory jokes and racist remarks through sentiment analysis and abuse detection from pre-trained and fine-tuned LLMs. Additionally, we curate and publish a "Hinduphobic COVID-19 X (Twitter) Dataset" of 8,000 tweets annotated for Hinduphobic abuse detection, which is used to fine-tune a BERT model, resulting in the development of the Hinduphobic BERT (HP-BERT) model. We then further fine-tune HP-BERT using the SenWave dataset for multi-label sentiment analysis. Our study encompasses approximately 27.4 million tweets from six countries, including Australia, Brazil, India, Indonesia, Japan, and the United Kingdom. Our findings reveal a strong correlation between spikes in COVID-19 cases and surges in Hinduphobic rhetoric, highlighting how political narratives, misinformation, and targeted jokes contributed to communal polarisation. These insights provide valuable guidance for developing strategies to mitigate communal tensions in future crises, both locally and globally. We advocate implementing automated monitoring and removal of such content on social media to curb divisive discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04156v1">AdaptiveCoPilot: Design and Testing of a NeuroAdaptive LLM Cockpit Guidance System in both Novice and Expert Pilots</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Pilots operating modern cockpits often face high cognitive demands due to complex interfaces and multitasking requirements, which can lead to overload and decreased performance. This study introduces AdaptiveCoPilot, a neuroadaptive guidance system that adapts visual, auditory, and textual cues in real time based on the pilot's cognitive workload, measured via functional Near-Infrared Spectroscopy (fNIRS). A formative study with expert pilots (N=3) identified adaptive rules for modality switching and information load adjustments during preflight tasks. These insights informed the design of AdaptiveCoPilot, which integrates cognitive state assessments, behavioral data, and adaptive strategies within a context-aware Large Language Model (LLM). The system was evaluated in a virtual reality (VR) simulated cockpit with licensed pilots (N=8), comparing its performance against baseline and random feedback conditions. The results indicate that the pilots using AdaptiveCoPilot exhibited higher rates of optimal cognitive load states on the facets of working memory and perception, along with reduced task completion times. Based on the formative study, experimental findings, qualitative interviews, we propose a set of strategies for future development of neuroadaptive pilot guidance systems and highlight the potential of neuroadaptive systems to enhance pilot performance and safety in aviation environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17218v2">To Err is Machine: Vulnerability Detection Challenges LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      In this paper, we present a challenging code reasoning task: vulnerability detection. Large Language Models (LLMs) have shown promising results in natural-language and math reasoning, but state-of-the-art (SOTA) models reported only 54.5% Balanced Accuracy in our vulnerability detection evaluation, even those models pre-trained on large amounts of source code. Our error analysis on LLM responses shows that the models struggle to reason about the code semantics relevant to identifying vulnerabilities, especially subtle semantic differences caused by small textual changes. We explored prominent models and training settings to understand their effects on vulnerability detection performance -- including better prompts, larger models, more pre-training data, and fine-tuning -- but none led to significant improvements. This raises the question of whether simply scaling training data and model size will allow us to "solve" complex code reasoning tasks like vulnerability detection, or if a fundamental shift in modeling and training techniques is required. We also explored adding domain knowledge to prompts; although it helped certain models understand some code semantics, vulnerability detection requires multi-step reasoning, and these models still failed in steps, such as reasoning about variable relations. Our results suggest that new models, new training methods, or more execution-specific pretraining data may be needed to conquer vulnerability detection. We speculate that auto-regressive pre-training on source code may not effectively extract code semantics, especially on the current pretraining mixtures, in which execution data is scarce. Success on vulnerability detection as a code reasoning task can benefit many areas of software engineering such as debugging, test input generation, and program repair. Our code and data are available at https://doi.org/10.6084/m9.figshare.27368025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04138v1">"Yeah Right!" -- Do LLMs Exhibit Multimodal Feature Transfer?</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Human communication is a multifaceted and multimodal skill. Communication requires an understanding of both the surface-level textual content and the connotative intent of a piece of communication. In humans, learning to go beyond the surface level starts by learning communicative intent in speech. Once humans acquire these skills in spoken communication, they transfer those skills to written communication. In this paper, we assess the ability of speech+text models and text models trained with special emphasis on human-to-human conversations to make this multimodal transfer of skill. We specifically test these models on their ability to detect covert deceptive communication. We find that with no special prompting speech+text LLMs have an advantage over unimodal LLMs in performing this task. Likewise, we find that human-to-human conversation-trained LLMs are also advantaged in this skill.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03991v1">Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 24 pages, 11 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Calibration, the alignment between model confidence and prediction accuracy, is critical for the reliable deployment of large language models (LLMs). Existing works neglect to measure the generalization of their methods to other prompt styles and different sizes of LLMs. To address this, we define a controlled experimental setting covering 12 LLMs and four prompt styles. We additionally investigate if incorporating the response agreement of multiple LLMs and an appropriate loss function can improve calibration performance. Concretely, we build Calib-n, a novel framework that trains an auxiliary model for confidence estimation that aggregates responses from multiple LLMs to capture inter-model agreement. To optimize calibration, we integrate focal and AUC surrogate losses alongside binary cross-entropy. Experiments across four datasets demonstrate that both response agreement and focal loss improve calibration from baselines. We find that few-shot prompts are the most effective for auxiliary model-based methods, and auxiliary models demonstrate robust calibration performance across accuracy variations, outperforming LLMs' internal probabilities and verbalized confidences. These insights deepen the understanding of influence factors in LLM calibration, supporting their reliable deployment in diverse applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13510v2">Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) workloads have distinct prefill and decode phases with different compute and memory requirements which should ideally be accounted for when scheduling input queries across different LLM instances in a cluster. However existing scheduling algorithms treat LLM workloads as monolithic jobs without considering the distinct characteristics of the two phases in each workload. This leads to sub-optimal scheduling and increased response latency. In this work, we start by characterizing factors affecting the response latency during LLM inference serving. We establish that better load balancing of inference requests across the available LLM instances can improve the end-to-end latency to a larger extent than merely focusing on optimizing the instance-level scheduler. Motivated by our findings, we propose a heuristic-guided reinforcement learning-based intelligent router for data-driven and workload-aware scheduling. Our router schedules queries across LLM instances by leveraging a trainable response-length predictor, and a novel formulation for estimating the impact of mixing different workloads and achieves over 11% lower end-to-end latency than existing approaches on a mix of public datasets and 7.8% lower end-to-end latency on real workload data with diverse input and output trends from Cloud Provider X. Additionally, the proposed framework can also serve as a standard for benchmarking different LLM inference schedulers since it provides the best latency for a given model, hardware, and instance-level scheduler combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14746v3">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale more than four fold (2) for efficient LLMs, the number of parameters N and the size D of a natural training corpus scale as $N \propto D^{0.44}$; (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03884v1">AlphaPO -- Reward shape matters for LLM alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 Preprint. Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) and its variants have made huge strides toward the effective alignment of large language models (LLMs) to follow instructions and reflect human values. More recently, Direct Alignment Algorithms (DAAs) have emerged in which the reward modeling stage of RLHF is skipped by characterizing the reward directly as a function of the policy being learned. Examples include Direct Preference Optimization (DPO) and Simple Preference Optimization (SimPO). These methods often suffer from likelihood displacement, a phenomenon by which the probabilities of preferred responses are often reduced undesirably. In this paper, we argue that, for DAAs the reward (function) shape matters. We introduce AlphaPO, a new DAA method that leverages an $\alpha$-parameter to help change the shape of the reward function beyond the standard log reward. AlphaPO helps maintain fine-grained control over likelihood displacement and over-optimization. Compared to SimPO, one of the best performing DAAs, AlphaPO leads to about 7\% to 10\% relative improvement in alignment performance for the instruct versions of Mistral-7B and Llama3-8B. The analysis and results presented highlight the importance of the reward shape, and how one can systematically change it to affect training dynamics, as well as improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14841v2">Helping LLMs Improve Code Generation Using Feedback from Testing and Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are one of the most promising developments in the field of artificial intelligence, and the software engineering community has readily noticed their potential role in the software development life-cycle. Developers routinely ask LLMs to generate code snippets, increasing productivity but also potentially introducing ownership, privacy, correctness, and security issues. Previous work highlighted how code generated by mainstream commercial LLMs is often not safe, containing vulnerabilities, bugs, and code smells. In this paper, we present a framework that leverages testing and static analysis to assess the quality, and guide the self-improvement, of code generated by general-purpose, open-source LLMs. First, we ask LLMs to generate C code to solve a number of programming tasks. Then we employ ground-truth tests to assess the (in)correctness of the generated code, and a static analysis tool to detect potential safety vulnerabilities. Next, we assess the models ability to evaluate the generated code, by asking them to detect errors and vulnerabilities. Finally, we test the models ability to fix the generated code, providing the reports produced during the static analysis and incorrectness evaluation phases as feedback. Our results show that models often produce incorrect code, and that the generated code can include safety issues. Moreover, they perform very poorly at detecting either issue. On the positive side, we observe a substantial ability to fix flawed code when provided with information about failed tests or potential vulnerabilities, indicating a promising avenue for improving the safety of LLM-based code generation tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19155v3">Lived Experience Not Found: LLMs Struggle to Align with Experts on Addressing Adverse Drug Reactions from Psychiatric Medication Use</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 30 pages, 8 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Adverse Drug Reactions (ADRs) from psychiatric medications are the leading cause of hospitalizations among mental health patients. With healthcare systems and online communities facing limitations in resolving ADR-related issues, Large Language Models (LLMs) have the potential to fill this gap. Despite the increasing capabilities of LLMs, past research has not explored their capabilities in detecting ADRs related to psychiatric medications or in providing effective harm reduction strategies. To address this, we introduce the Psych-ADR benchmark and the Adverse Drug Reaction Response Assessment (ADRA) framework to systematically evaluate LLM performance in detecting ADR expressions and delivering expert-aligned mitigation strategies. Our analyses show that LLMs struggle with understanding the nuances of ADRs and differentiating between types of ADRs. While LLMs align with experts in terms of expressed emotions and tone of the text, their responses are more complex, harder to read, and only 70.86% aligned with expert strategies. Furthermore, they provide less actionable advice by a margin of 12.32% on average. Our work provides a comprehensive benchmark and evaluation framework for assessing LLMs in strategy-driven tasks within high-risk domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17481v2">A Survey on LLM-based Multi-Agent System: Recent Advances and New Frontiers in Application</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 13 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      LLM-based Multi-Agent Systems ( LLM-MAS ) have become a research hotspot since the rise of large language models (LLMs). However, with the continuous influx of new related works, the existing reviews struggle to capture them comprehensively. This paper presents a comprehensive survey of these studies. We first discuss the definition of LLM-MAS, a framework encompassing much of previous work. We provide an overview of the various applications of LLM-MAS in (i) solving complex tasks, (ii) simulating specific scenarios, and (iii) evaluating generative agents. Building on previous studies, we also highlight several challenges and propose future directions for research in this field.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11876v2">Rescriber: Smaller-LLM-Powered User-Led Data Minimization for Navigating Privacy Trade-offs in LLM-Based Conversational Agent</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      The proliferation of LLM-based conversational agents has resulted in excessive disclosure of identifiable or sensitive information. However, existing technologies fail to offer perceptible control or account for users' personal preferences about privacy-utility tradeoffs due to the lack of user involvement. To bridge this gap, we designed, built, and evaluated Rescriber, a browser extension that supports user-led data minimization in LLM-based conversational agents by helping users detect and sanitize personal information in their prompts. Our studies (N=12) showed that Rescriber helped users reduce unnecessary disclosure and addressed their privacy concerns. Users' subjective perceptions of the system powered by Llama3-8B were on par with that by GPT-4o. The comprehensiveness and consistency of the detection and sanitization emerge as essential factors that affect users' trust and perceived protection. Our findings confirm the viability of smaller-LLM-powered, user-facing, on-device privacy controls, presenting a promising approach to address the privacy and trust challenges of AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12935v2">ChatBug: A Common Vulnerability of Aligned LLMs Induced by Chat Templates</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 This paper is accepted to AAAI 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are expected to follow instructions from users and engage in conversations. Techniques to enhance LLMs' instruction-following capabilities typically fine-tune them using data structured according to a predefined chat template. Although chat templates are shown to be effective in optimizing LLM performance, their impact on safety alignment of LLMs has been less understood, which is crucial for deploying LLMs safely at scale. In this paper, we investigate how chat templates affect safety alignment of LLMs. We identify a common vulnerability, named ChatBug, that is introduced by chat templates. Our key insight to identify ChatBug is that the chat templates provide a rigid format that need to be followed by LLMs, but not by users. Hence, a malicious user may not necessarily follow the chat template when prompting LLMs. Instead, malicious users could leverage their knowledge of the chat template and accordingly craft their prompts to bypass safety alignments of LLMs. We develop two attacks to exploit the ChatBug vulnerability. We demonstrate that a malicious user can exploit the ChatBug vulnerability of eight state-of-the-art (SOTA) LLMs and effectively elicit unintended responses from these models. Moreover, we show that ChatBug can be exploited by existing jailbreak attacks to enhance their attack success rates. We investigate potential countermeasures to ChatBug. Our results show that while adversarial training effectively mitigates the ChatBug vulnerability, the victim model incurs significant performance degradation. These results highlight the trade-off between safety alignment and helpfulness. Developing new methods for instruction tuning to balance this trade-off is an open and critical direction for future research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.17740v2">Bridging the Language Gap: Dynamic Learning Strategies for Improving Multilingual Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized various domains but still struggle with non-Latin scripts and low-resource languages. This paper addresses the critical challenge of improving multilingual performance without extensive fine-tuning. We introduce a novel dynamic learning approach that optimizes prompt strategy, embedding model, and LLM per query at runtime. By adapting configurations dynamically, our method achieves significant improvements over static, best and random baselines. It operates efficiently in both offline and online settings, generalizing seamlessly across new languages and datasets. Leveraging Retrieval-Augmented Generation (RAG) with state-of-the-art multilingual embeddings, we achieve superior task performance across diverse linguistic contexts. Through systematic investigation and evaluation across 18 diverse languages using popular question-answering (QA) datasets we show our approach results in 10-15% improvements in multilingual performance over pre-trained models and 4x gains compared to fine-tuned, language-specific models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00166v2">EEG Emotion Copilot: Optimizing Lightweight LLMs for Emotional EEG Interpretation with Assisted Medical Record Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 10 pages, 12 figures, 2 tables
    </div>
    <details class="paper-abstract">
      In the fields of affective computing (AC) and brain-machine interface (BMI), the analysis of physiological and behavioral signals to discern individual emotional states has emerged as a critical research frontier. While deep learning-based approaches have made notable strides in EEG emotion recognition, particularly in feature extraction and pattern recognition, significant challenges persist in achieving end-to-end emotion computation, including real-time processing, individual adaptation, and seamless user interaction. This paper presents the EEG Emotion Copilot, a system optimizing a lightweight large language model (LLM) with 0.5B parameters operating in a local setting, which first recognizes emotional states directly from EEG signals, subsequently generates personalized diagnostic and treatment suggestions, and finally supports the automation of assisted electronic medical records. Specifically, we demonstrate the critical techniques in the novel data structure of prompt, model pruning and fine-tuning training, and deployment strategies aiming at improving real-time performance and computational efficiency. Extensive experiments show that our optimized lightweight LLM-based copilot achieves an enhanced intuitive interface for participant interaction, superior accuracy of emotion recognition and assisted electronic medical records generation, in comparison to such models with similar scale parameters or large-scale parameters such as 1.5B, 1.8B, 3B and 7B. In summary, through these efforts, the proposed copilot is expected to advance the application of AC in the medical domain, offering innovative solution to mental health monitoring. The codes will be released at https://github.com/NZWANG/EEG_Emotion_Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03491v1">Can LLMs Design Good Questions Based on Context?</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      This paper evaluates questions generated by LLMs from context, comparing them to human-generated questions across six dimensions. We introduce an automated LLM-based evaluation method, focusing on aspects like question length, type, context coverage, and answerability. Our findings highlight unique characteristics of LLM-generated questions, contributing insights that can support further research in question quality and downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03486v1">Align-Pro: A Principled Approach to Prompt Optimization for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-07
      | 💬 27 pages, Accepted in AAAI 2025
    </div>
    <details class="paper-abstract">
      The alignment of large language models (LLMs) with human values is critical as these models become increasingly integrated into various societal and decision-making processes. Traditional methods, such as reinforcement learning from human feedback (RLHF), achieve alignment by fine-tuning model parameters, but these approaches are often computationally expensive and impractical when models are frozen or inaccessible for parameter modification. In contrast, prompt optimization is a viable alternative to RLHF for LLM alignment. While the existing literature has shown empirical promise of prompt optimization, its theoretical underpinning remains under-explored. We address this gap by formulating prompt optimization as an optimization problem and try to provide theoretical insights into the optimality of such a framework. To analyze the performance of the prompt optimization, we study theoretical suboptimality bounds and provide insights in terms of how prompt optimization depends upon the given prompter and target model. We also provide empirical validation through experiments on various datasets, demonstrating that prompt optimization can effectively align LLMs, even when parameter fine-tuning is not feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22376v2">Rare-to-Frequent: Unlocking Compositional Generation Power of Diffusion Models on Rare Concepts with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      State-of-the-art text-to-image (T2I) diffusion models often struggle to generate rare compositions of concepts, e.g., objects with unusual attributes. In this paper, we show that the compositional generation power of diffusion models on such rare concepts can be significantly enhanced by the Large Language Model (LLM) guidance. We start with empirical and theoretical analysis, demonstrating that exposing frequent concepts relevant to the target rare concepts during the diffusion sampling process yields more accurate concept composition. Based on this, we propose a training-free approach, R2F, that plans and executes the overall rare-to-frequent concept guidance throughout the diffusion inference by leveraging the abundant semantic knowledge in LLMs. Our framework is flexible across any pre-trained diffusion models and LLMs, and can be seamlessly integrated with the region-guided diffusion approaches. Extensive experiments on three datasets, including our newly proposed benchmark, RareBench, containing various prompts with rare compositions of concepts, R2F significantly surpasses existing models including SD3.0 and FLUX by up to 28.1%p in T2I alignment. Code is available at https://github.com/krafton-ai/Rare-to-Frequent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00826v2">LLM-Powered Multi-Agent System for Automated Crypto Portfolio Management</a></div>
    <div class="paper-meta">
      📅 2025-01-07
    </div>
    <details class="paper-abstract">
      Cryptocurrency investment is inherently difficult due to its shorter history compared to traditional assets, the need to integrate vast amounts of data from various modalities, and the requirement for complex reasoning. While deep learning approaches have been applied to address these challenges, their black-box nature raises concerns about trust and explainability. Recently, large language models (LLMs) have shown promise in financial applications due to their ability to understand multi-modal data and generate explainable decisions. However, single LLM faces limitations in complex, comprehensive tasks such as asset investment. These limitations are even more pronounced in cryptocurrency investment, where LLMs have less domain-specific knowledge in their training corpora. To overcome these challenges, we propose an explainable, multi-modal, multi-agent framework for cryptocurrency investment. Our framework uses specialized agents that collaborate within and across teams to handle subtasks such as data analysis, literature integration, and investment decision-making for the top 30 cryptocurrencies by market capitalization. The expert training module fine-tunes agents using multi-modal historical data and professional investment literature, while the multi-agent investment module employs real-time data to make informed cryptocurrency investment decisions. Unique intrateam and interteam collaboration mechanisms enhance prediction accuracy by adjusting final predictions based on confidence levels within agent teams and facilitating information sharing between teams. Empirical evaluation using data from November 2023 to September 2024 demonstrates that our framework outperforms single-agent models and market benchmarks in classification, asset pricing, portfolio, and explainability performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13226v2">Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly trained on datasets consisting of fixed-length token sequences. These datasets are created by randomly concatenating documents of various lengths and then chunking them into sequences of a predetermined target length (concat-and-chunk). Recent attention implementations mask cross-document attention, reducing the effective length of a chunk of tokens. Additionally, training on long sequences becomes computationally prohibitive due to the quadratic cost of attention. In this study, we introduce dataset decomposition, a novel variable sequence length training technique, to tackle these challenges. We decompose a dataset into a union of buckets, each containing sequences of the same size extracted from a unique document. During training, we use variable sequence length and batch-size, sampling simultaneously from all buckets with a curriculum. In contrast to the concat-and-chunk baseline, which incurs a fixed attention cost at every step of training, our proposed method incurs a computational cost proportional to the actual document lengths at each step, resulting in significant savings in training time. We train an 8k context-length 1B model at the same cost as a 2k context-length model trained with the baseline approach. Experiments on a web-scale corpus demonstrate that our approach significantly enhances performance on standard language evaluations and long-context benchmarks, reaching target accuracy with up to 6x faster training compared to the baseline. Our method not only enables efficient pretraining on long sequences but also scales effectively with dataset size. Lastly, we shed light on a critical yet less studied aspect of training large language models: the distribution and curriculum of sequence lengths, which results in a non-negligible difference in performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00168v2">Takeaways from Applying LLM Capabilities to Multiple Conversational Avatars in a VR Pilot Study</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      We present a virtual reality (VR) environment featuring conversational avatars powered by a locally-deployed LLM, integrated with automatic speech recognition (ASR), text-to-speech (TTS), and lip-syncing. Through a pilot study, we explored the effects of three types of avatar status indicators during response generation. Our findings reveal design considerations for improving responsiveness and realism in LLM-driven conversational systems. We also detail two system architectures: one using an LLM-based state machine to control avatar behavior and another integrating retrieval-augmented generation (RAG) for context-grounded responses. Together, these contributions offer practical insights to guide future work in developing task-oriented conversational AI in VR environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03218v1">Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      Active Real-time interaction with video LLMs introduces a new paradigm for human-computer interaction, where the model not only understands user intent but also responds while continuously processing streaming video on the fly. Unlike offline video LLMs, which analyze the entire video before answering questions, active real-time interaction requires three capabilities: 1) Perception: real-time video monitoring and interaction capturing. 2) Decision: raising proactive interaction in proper situations, 3) Reaction: continuous interaction with users. However, inherent conflicts exist among the desired capabilities. The Decision and Reaction require a contrary Perception scale and grain, and the autoregressive decoding blocks the real-time Perception and Decision during the Reaction. To unify the conflicted capabilities within a harmonious system, we present Dispider, a system that disentangles Perception, Decision, and Reaction. Dispider features a lightweight proactive streaming video processing module that tracks the video stream and identifies optimal moments for interaction. Once the interaction is triggered, an asynchronous interaction module provides detailed responses, while the processing module continues to monitor the video in the meantime. Our disentangled and asynchronous design ensures timely, contextually accurate, and computationally efficient responses, making Dispider ideal for active real-time interaction for long-duration video streams. Experiments show that Dispider not only maintains strong performance in conventional video QA tasks, but also significantly surpasses previous online models in streaming scenario responses, thereby validating the effectiveness of our architecture. The code and model are released at \url{https://github.com/Mark12Ding/Dispider}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03212v1">Leveraging Explainable AI for LLM Text Attribution: Differentiating Human-Written and Multiple LLMs-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The development of Generative AI Large Language Models (LLMs) raised the alarm regarding identifying content produced through generative AI or humans. In one case, issues arise when students heavily rely on such tools in a manner that can affect the development of their writing or coding skills. Other issues of plagiarism also apply. This study aims to support efforts to detect and identify textual content generated using LLM tools. We hypothesize that LLMs-generated text is detectable by machine learning (ML), and investigate ML models that can recognize and differentiate texts generated by multiple LLMs tools. We leverage several ML and Deep Learning (DL) algorithms such as Random Forest (RF), and Recurrent Neural Networks (RNN), and utilized Explainable Artificial Intelligence (XAI) to understand the important features in attribution. Our method is divided into 1) binary classification to differentiate between human-written and AI-text, and 2) multi classification, to differentiate between human-written text and the text generated by the five different LLM tools (ChatGPT, LLaMA, Google Bard, Claude, and Perplexity). Results show high accuracy in the multi and binary classification. Our model outperformed GPTZero with 98.5\% accuracy to 78.3\%. Notably, GPTZero was unable to recognize about 4.2\% of the observations, but our model was able to recognize the complete test dataset. XAI results showed that understanding feature importance across different classes enables detailed author/source profiles. Further, aiding in attribution and supporting plagiarism detection by highlighting unique stylistic and structural elements ensuring robust content originality verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16353v2">The Two-Hop Curse: LLMs trained on A$\rightarrow$B, B$\rightarrow$C fail to learn A$\rightarrow$C</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      [Notice: This version is outdated. Recent research contradicts some key claims; we are working on a major revision with more nuanced analysis. Please wait for the updated version.] While LLMs excel at multi-hop questions (e.g. "Who is the spouse of the performer of Imagine?") when using chain-of-thought reasoning (CoT), they struggle when forced to reason internally (without CoT). Previous work on the size and nature of this gap produced mixed evidence with inconclusive results. In this paper, we introduce a controlled setting for investigating two-hop reasoning in LLMs, where the above-chance performance constitutes undeniable evidence for latent reasoning. We fine-tune LLMs (including Llama 3 8B Instruct and GPT-4o) on fictional facts and confirm that they generalize to answering two-hop questions about them using CoT. We find that models can perform latent reasoning when facts appear together during training or in the prompt. However, to our surprise, models completely fail at two-hop reasoning without CoT when learned facts only appear in different documents, achieving chance-level accuracy and chance-level test loss. We call this complete failure to compose separately learned facts the Two-Hop Curse. Moreover, we evaluate 9 frontier LLMs on real-world facts, finding that models completely fail at two-hop no-CoT reasoning for over half of question categories while maintaining partial success with CoT across most categories. These results suggest that LLMs lack a general capability for latent multi-hop reasoning independent of the question type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13147v3">Are Your LLMs Capable of Stable Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has demonstrated remarkable progress in complex reasoning tasks. However, a significant discrepancy persists between benchmark performances and real-world applications. We identify this gap as primarily stemming from current evaluation protocols and metrics, which inadequately capture the full spectrum of LLM capabilities, particularly in complex reasoning tasks where both accuracy and consistency are crucial. This work makes two key contributions. First, we introduce G-Pass@k, a novel evaluation metric that provides a continuous assessment of model performance across multiple sampling attempts, quantifying both the model's peak performance potential and its stability. Second, we present LiveMathBench, a dynamic benchmark comprising challenging, contemporary mathematical problems designed to minimize data leakage risks during evaluation. Through extensive experiments using G-Pass@k on state-of-the-art LLMs with LiveMathBench, we provide comprehensive insights into both their maximum capabilities and operational consistency. Our findings reveal substantial room for improvement in LLMs' "realistic" reasoning capabilities, highlighting the need for more robust evaluation methods. The benchmark and detailed results are available at: https://github.com/open-compass/GPassK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.03628v2">Synthetic Oversampling: Theory and A Practical Approach Using LLMs to Address Data Imbalance</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 82 pages, 28 figures
    </div>
    <details class="paper-abstract">
      Imbalanced classification and spurious correlation are common challenges in data science and machine learning. Both issues are linked to data imbalance, with certain groups of data samples significantly underrepresented, which in turn would compromise the accuracy, robustness and generalizability of the learned models. Recent advances have proposed leveraging the flexibility and generative capabilities of large language models (LLMs), typically built on transformer architectures, to generate synthetic samples and to augment the observed data. In the context of imbalanced data, LLMs are used to oversample underrepresented groups and have shown promising improvements. However, there is a clear lack of theoretical understanding of such synthetic data approaches. In this article, we develop novel theoretical foundations to systematically study the roles of synthetic samples in addressing imbalanced classification and spurious correlation. Specifically, we first explicitly quantify the benefits of synthetic oversampling. Next, we analyze the scaling dynamics in synthetic data augmentation, and derive the corresponding scaling law. Finally, we demonstrate the capacity of transformer models to generate high-quality synthetic samples. We further conduct extensive numerical experiments to validate the efficacy of the LLM-based synthetic oversampling and augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00722v2">Pathway to Secure and Trustworthy ZSM for LLMs: Attacks, Defense, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been gaining a lot of interest due to their adaptability and extensibility in emerging applications, including communication networks. It is anticipated that ZSM networks will be able to support LLMs as a service, as they provide ultra reliable low-latency communications and closed loop massive connectivity. However, LLMs are vulnerable to data and model privacy issues that affect the trustworthiness of LLMs to be deployed for user-based services. In this paper, we explore the security vulnerabilities associated with fine-tuning LLMs in ZSM networks, in particular the membership inference attack. We define the characteristics of an attack network that can perform a membership inference attack if the attacker has access to the fine-tuned model for the downstream task. We show that the membership inference attacks are effective for any downstream task, which can lead to a personal data breach when using LLM as a service. The experimental results show that the attack success rate of maximum 92% can be achieved on named entity recognition task. Based on the experimental analysis, we discuss possible defense mechanisms and present possible research directions to make the LLMs more trustworthy in the context of ZSM networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09830v3">Automating the Generation of Prompts for LLM-based Action Choice in PDDL Planning</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Latest Version of the paper previously called "AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL"; Added extended experiments; newer gpt4 model
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized a large variety of NLP tasks. An active debate is to what extent they can do reasoning and planning. Prior work has assessed the latter in the specific context of PDDL planning, based on manually converting three PDDL domains into natural language (NL) prompts. Here we automate this conversion step, showing how to leverage an LLM to automatically generate NL prompts from PDDL input. Our automatically generated NL prompts result in similar LLM-planning performance as the previous manually generated ones. Beyond this, the automation enables us to run much larger experiments, providing for the first time a broad evaluation of LLM planning performance in PDDL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03012v1">Analyzing Fine-tuning Representation Shift for Multimodal LLMs Steering alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 The first three authors contributed equally
    </div>
    <details class="paper-abstract">
      Multimodal LLMs have reached remarkable levels of proficiency in understanding multimodal inputs, driving extensive research to develop increasingly powerful models. However, much less attention has been paid to understanding and explaining the underlying mechanisms of these models. Most existing explainability research examines these models only in their final states, overlooking the dynamic representational shifts that occur during training. In this work, we systematically analyze the evolution of hidden state representations to reveal how fine-tuning alters the internal structure of a model to specialize in new multimodal tasks. Using a concept-based approach, we map hidden states to interpretable visual and textual concepts, enabling us to trace changes in encoded concepts across modalities as training progresses. We also demonstrate the use of shift vectors to capture these concepts changes. These shift vectors allow us to recover fine-tuned concepts by shifting those in the original model. Finally, we explore the practical impact of our findings on model steering, showing that we can adjust multimodal LLMs behaviors without any training, such as modifying answer types, captions style, or biasing the model toward specific responses. Our work sheds light on how multimodal representations evolve through fine-tuning and offers a new perspective for interpreting model adaptation in multimodal tasks. The code for this project is publicly available at https://github.com/mshukor/xl-vlms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02885v1">MDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Video large language models (Video-LLMs) have made significant progress in understanding videos. However, processing multiple frames leads to lengthy visual token sequences, presenting challenges such as the limited context length cannot accommodate the entire video, and the inclusion of irrelevant frames hinders visual perception. Hence, effective frame selection is crucial. This paper emphasizes that frame selection should follow three key principles: query relevance, list-wise diversity, and sequentiality. Existing methods, such as uniform frame sampling and query-frame matching, do not capture all of these principles. Thus, we propose Markov decision determinantal point process with dynamic programming (MDP3) for frame selection, a training-free and model-agnostic method that can be seamlessly integrated into existing Video-LLMs. Our method first estimates frame similarities conditioned on the query using a conditional Gaussian kernel within the reproducing kernel Hilbert space~(RKHS). We then apply the determinantal point process~(DPP) to the similarity matrix to capture both query relevance and list-wise diversity. To incorporate sequentiality, we segment the video and apply DPP within each segment, conditioned on the preceding segment selection, modeled as a Markov decision process~(MDP) for allocating selection sizes across segments. Theoretically, MDP3 provides a \((1 - 1/e)\)-approximate solution to the NP-hard list-wise frame selection problem with pseudo-polynomial time complexity, demonstrating its efficiency. Empirically, MDP3 significantly outperforms existing methods, verifying its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20787v3">SecBench: A Comprehensive Multi-Dimensional Benchmarking Dataset for LLMs in Cybersecurity</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) is crucial for understanding their capabilities and limitations across various applications, including natural language processing and code generation. Existing benchmarks like MMLU, C-Eval, and HumanEval assess general LLM performance but lack focus on specific expert domains such as cybersecurity. Previous attempts to create cybersecurity datasets have faced limitations, including insufficient data volume and a reliance on multiple-choice questions (MCQs). To address these gaps, we propose SecBench, a multi-dimensional benchmarking dataset designed to evaluate LLMs in the cybersecurity domain. SecBench includes questions in various formats (MCQs and short-answer questions (SAQs)), at different capability levels (Knowledge Retention and Logical Reasoning), in multiple languages (Chinese and English), and across various sub-domains. The dataset was constructed by collecting high-quality data from open sources and organizing a Cybersecurity Question Design Contest, resulting in 44,823 MCQs and 3,087 SAQs. Particularly, we used the powerful while cost-effective LLMs to (1). label the data and (2). constructing a grading agent for automatic evaluation of SAQs. Benchmarking results on 16 SOTA LLMs demonstrate the usability of SecBench, which is arguably the largest and most comprehensive benchmark dataset for LLMs in cybersecurity. More information about SecBench can be found at our website, and the dataset can be accessed via the artifact link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03288v1">CodeVision: Detecting LLM-Generated Code Using 2D Token Probability Maps and Vision Models</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) like ChatGPT has significantly improved automated code generation, enhancing software development efficiency. However, this introduces challenges in academia, particularly in distinguishing between human-written and LLM-generated code, which complicates issues of academic integrity. Existing detection methods, such as pre-trained models and watermarking, face limitations in adaptability and computational efficiency. In this paper, we propose a novel detection method using 2D token probability maps combined with vision models, preserving spatial code structures such as indentation and brackets. By transforming code into log probability matrices and applying vision models like Vision Transformers (ViT) and ResNet, we capture both content and structure for more accurate detection. Our method shows robustness across multiple programming languages and improves upon traditional detectors, offering a scalable and computationally efficient solution for identifying LLM-generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16594v5">From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 v5: add new citations; 36 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Assessment and evaluation have long been critical challenges in artificial intelligence (AI) and natural language processing (NLP). However, traditional methods, whether matching-based or embedding-based, often fall short of judging subtle attributes and delivering satisfactory results. Recent advancements in Large Language Models (LLMs) inspire the "LLM-as-a-judge" paradigm, where LLMs are leveraged to perform scoring, ranking, or selection across various tasks and applications. This paper provides a comprehensive survey of LLM-based judgment and assessment, offering an in-depth overview to advance this emerging field. We begin by giving detailed definitions from both input and output perspectives. Then we introduce a comprehensive taxonomy to explore LLM-as-a-judge from three dimensions: what to judge, how to judge and where to judge. Finally, we compile benchmarks for evaluating LLM-as-a-judge and highlight key challenges and promising directions, aiming to provide valuable insights and inspire future research in this promising research area. Paper list and more resources about LLM-as-a-judge can be found at \url{https://github.com/llm-as-a-judge/Awesome-LLM-as-a-judge} and \url{https://llm-as-a-judge.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12470v3">DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 Accepted by WSDM 2025
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into recommender systems has led to substantial performance improvements. However, this often comes at the cost of diminished recommendation diversity, which can negatively impact user satisfaction. To address this issue, controllable recommendation has emerged as a promising approach, allowing users to specify their preferences and receive recommendations that meet their diverse needs. Despite its potential, existing controllable recommender systems frequently rely on simplistic mechanisms, such as a single prompt, to regulate diversity-an approach that falls short of capturing the full complexity of user preferences. In response to these limitations, we propose DLCRec, a novel framework designed to enable fine-grained control over diversity in LLM-based recommendations. Unlike traditional methods, DLCRec adopts a fine-grained task decomposition strategy, breaking down the recommendation process into three sequential sub-tasks: genre prediction, genre filling, and item prediction. These sub-tasks are trained independently and inferred sequentially according to user-defined control numbers, ensuring more precise control over diversity. Furthermore, the scarcity and uneven distribution of diversity-related user behavior data pose significant challenges for fine-tuning. To overcome these obstacles, we introduce two data augmentation techniques that enhance the model's robustness to noisy and out-of-distribution data. These techniques expose the model to a broader range of patterns, improving its adaptability in generating recommendations with varying levels of diversity. Our extensive empirical evaluation demonstrates that DLCRec not only provides precise control over diversity but also outperforms state-of-the-art baselines across multiple recommendation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14743v2">LlamaPartialSpoof: An LLM-Driven Fake Speech Dataset Simulating Disinformation Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-06
      | 💬 5 pages, ICASSP 2025
    </div>
    <details class="paper-abstract">
      Previous fake speech datasets were constructed from a defender's perspective to develop countermeasure (CM) systems without considering diverse motivations of attackers. To better align with real-life scenarios, we created LlamaPartialSpoof, a 130-hour dataset that contains both fully and partially fake speech, using a large language model (LLM) and voice cloning technologies to evaluate the robustness of CMs. By examining valuable information for both attackers and defenders, we identify several key vulnerabilities in current CM systems, which can be exploited to enhance attack success rates, including biases toward certain text-to-speech models or concatenation methods. Our experimental results indicate that the current fake speech detection system struggle to generalize to unseen scenarios, achieving a best performance of 24.49% equal error rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09824v6">LLM-Based Multi-Agent Systems are Scalable Graph Generative Models</a></div>
    <div class="paper-meta">
      📅 2025-01-06
    </div>
    <details class="paper-abstract">
      The structural properties of naturally arising social graphs are extensively studied to understand their evolution. Prior approaches for modeling network dynamics typically rely on rule-based models, which lack realism and generalizability, or deep learning-based models, which require large-scale training datasets. Social graphs, as abstract graph representations of entity-wise interactions, present an opportunity to explore network evolution mechanisms through realistic simulations of human-item interactions. Leveraging the pre-trained social consensus knowledge embedded in large language models (LLMs), we present GraphAgent-Generator (GAG), a novel simulation-based framework for dynamic, text-attributed social graph generation. GAG simulates the temporal node and edge generation processes for zero-shot social graph generation. The resulting graphs exhibit adherence to seven key macroscopic network properties, achieving an 11% improvement in microscopic graph structure metrics. Through the node classification benchmarking task, we validate GAG effectively captures the intricate text-structure correlations in graph generation. Furthermore, GAG supports generating graphs with up to nearly 100,000 nodes or 10 million edges through large-scale LLM-based agent simulation with parallel acceleration, achieving a minimum speed-up of 90.4%. The source code is available at https://github.com/Ji-Cather/GraphAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02688v1">Decoding specialised feature neurons in LLMs with the final projection layer</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 5 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) typically have billions of parameters and are thus often difficult to interpret in their operation. Such black-box models can pose a significant risk to safety when trusted to make important decisions. The lack of interpretability of LLMs is more related to their sheer size, rather than the complexity of their individual components. The TARS method for knowledge removal (Davies et al 2024) provides strong evidence for the hypothesis that that linear layer weights which act directly on the residual stream may have high correlation with different concepts encoded in the residual stream. Building upon this, we attempt to decode neuron weights directly into token probabilities through the final projection layer of the model (the LM-head). Firstly, we show that with Llama 3.1 8B we can utilise the LM-head to decode specialised feature neurons that respond strongly to certain concepts, with examples such as "dog" and "California". This is then confirmed by demonstrating that these neurons can be clamped to affect the probability of the concept in the output. This extends to the fine-tuned assistant Llama 3.1 8B instruct model, where we find that over 75% of neurons in the up-projection layers have the same top associated token compared to the pretrained model. Finally, we demonstrate that clamping the "dog" neuron leads the instruct model to always discuss dogs when asked about its favourite animal. Through our method, it is possible to map the entirety of Llama 3.1 8B's up-projection neurons in less than 15 minutes with no parallelization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03516v3">Query3D: LLM-Powered Open-Vocabulary Scene Segmentation with Language Embedded 3D Gaussian</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      This paper introduces a novel method for open-vocabulary 3D scene querying in autonomous driving by combining Language Embedded 3D Gaussians with Large Language Models (LLMs). We propose utilizing LLMs to generate both contextually canonical phrases and helping positive words for enhanced segmentation and scene interpretation. Our method leverages GPT-3.5 Turbo as an expert model to create a high-quality text dataset, which we then use to fine-tune smaller, more efficient LLMs for on-device deployment. Our comprehensive evaluation on the WayveScenes101 dataset demonstrates that LLM-guided segmentation significantly outperforms traditional approaches based on predefined canonical phrases. Notably, our fine-tuned smaller models achieve performance comparable to larger expert models while maintaining faster inference times. Through ablation studies, we discover that the effectiveness of helping positive words correlates with model scale, with larger models better equipped to leverage additional semantic information. This work represents a significant advancement towards more efficient, context-aware autonomous driving systems, effectively bridging 3D scene representation with high-level semantic querying while maintaining practical deployment considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02628v1">Cracks in The Stack: Hidden Vulnerabilities and Licensing Risks in LLM Pre-Training Datasets</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 Accepted in the Second International Workshop on Large Language Models for Code (LLM4Code 2025)
    </div>
    <details class="paper-abstract">
      A critical part of creating code suggestion systems is the pre-training of Large Language Models on vast amounts of source code and natural language text, often of questionable origin or quality. This may contribute to the presence of bugs and vulnerabilities in code generated by LLMs. While efforts to identify bugs at or after code generation exist, it is preferable to pre-train or fine-tune LLMs on curated, high-quality, and compliant datasets. The need for vast amounts of training data necessitates that such curation be automated, minimizing human intervention. We propose an automated source code autocuration technique that leverages the complete version history of open-source software projects to improve the quality of training data. This approach leverages the version history of all OSS projects to identify training data samples that have been modified or have undergone changes in at least one OSS project, and pinpoint a subset of samples that include fixes for bugs or vulnerabilities. We evaluate this method using The Stack v2 dataset, and find that 17% of the code versions in the dataset have newer versions, with 17% of those representing bug fixes, including 2.36% addressing known CVEs. The deduplicated version of Stack v2 still includes blobs vulnerable to 6,947 known CVEs. Furthermore, 58% of the blobs in the dataset were never modified after creation, suggesting they likely represent software with minimal or no use. Misidentified blob origins present an additional challenge, as they lead to the inclusion of non-permissively licensed code, raising serious compliance concerns. By addressing these issues, the training of new models can avoid perpetuating buggy code patterns or license violations. We expect our results to inspire process improvements for automated data curation, with the potential to enhance the reliability of outputs generated by AI tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02625v1">HALO: Hadamard-Assisted Lossless Optimization for Efficient Low-Precision LLM Training and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Quantized training of Large Language Models (LLMs) remains an open challenge, as maintaining accuracy while performing all matrix multiplications in low precision has proven difficult. This is particularly the case when fine-tuning pre-trained models, which often already have large weight and activation outlier values that render quantized optimization difficult. We present HALO, a novel quantization-aware training approach for Transformers that enables accurate and efficient low-precision training by combining 1) strategic placement of Hadamard rotations in both forward and backward passes, to mitigate outliers during the low-precision computation, 2) FSDP integration for low-precision communication, and 3) high-performance kernel support. Our approach ensures that all large matrix multiplications during the forward and backward passes are executed in lower precision. Applied to LLAMA-family models, HALO achieves near-full-precision-equivalent results during fine-tuning on various tasks, while delivering up to 1.31x end-to-end speedup for full fine-tuning on RTX 4090 GPUs. Our method supports both standard and parameter-efficient fine-tuning (PEFT) methods, both backed by efficient kernel implementations. Our results demonstrate the first practical approach to fully quantized LLM fine-tuning that maintains accuracy in FP8 precision, while delivering performance benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02621v1">LLMs Help Alleviate the Cross-Subject Variability in Brain Signal and Language Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      Decoding human activity from EEG signals has long been a popular research topic. While recent studies have increasingly shifted focus from single-subject to cross-subject analysis, few have explored the model's ability to perform zero-shot predictions on EEG signals from previously unseen subjects. This research aims to investigate whether deep learning methods can capture subject-independent semantic information inherent in human EEG signals. Such insights are crucial for Brain-Computer Interfaces (BCI) because, on one hand, they demonstrate the model's robustness against subject-specific temporal biases, and on the other, they significantly enhance the generalizability of downstream tasks. We employ Large Language Models (LLMs) as denoising agents to extract subject-independent semantic features from noisy EEG signals. Experimental results, including ablation studies, highlight the pivotal role of LLMs in decoding subject-independent semantic information from noisy EEG data. We hope our findings will contribute to advancing BCI research and assist both academia and industry in applying EEG signals to a broader range of applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10351v3">Bias Unveiled: Investigating Social Bias in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-01-05
      | 💬 9pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the field of automated code generation. However, a notable research gap exists in the evaluation of social biases that may be present in the code produced by LLMs. To solve this issue, we propose a novel fairness framework, i.e., Solar, to assess and mitigate the social biases of LLM-generated code. Specifically, Solar can automatically generate test cases for quantitatively uncovering social biases of the auto-generated code by LLMs. To quantify the severity of social biases in generated code, we develop a dataset that covers a diverse set of social problems. We applied Solar and the crafted dataset to four state-of-the-art LLMs for code generation. Our evaluation reveals severe bias in the LLM-generated code from all the subject LLMs. Furthermore, we explore several strategies for bias mitigation, including Chain-of-Thought (CoT) prompting, combining positive role-playing with CoT prompting and iterative prompting. Our experiments show that iterative prompting can effectively reduce social bias in LLM-generated code by up to 90%. Solar is highly extensible to evaluate new social problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02600v1">TAPAS: Thermal- and Power-Aware Scheduling for LLM Inference in Cloud Platforms</a></div>
    <div class="paper-meta">
      📅 2025-01-05
    </div>
    <details class="paper-abstract">
      The rising demand for generative large language models (LLMs) poses challenges for thermal and power management in cloud datacenters. Traditional techniques often are inadequate for LLM inference due to the fine-grained, millisecond-scale execution phases, each with distinct performance, thermal, and power profiles. Additionally, LLM inference workloads are sensitive to various configuration parameters (e.g., model parallelism, size, and quantization) that involve trade-offs between performance, temperature, power, and output quality. Moreover, clouds often co-locate SaaS and IaaS workloads, each with different levels of visibility and flexibility. We propose TAPAS, a thermal- and power-aware framework designed for LLM inference clusters in the cloud. TAPAS enhances cooling and power oversubscription capabilities, reducing the total cost of ownership (TCO) while effectively handling emergencies (e.g., cooling and power failures). The system leverages historical temperature and power data, along with the adaptability of SaaS workloads, to: (1) efficiently place new GPU workload VMs within cooling and power constraints, (2) route LLM inference requests across SaaS VMs, and (3) reconfigure SaaS VMs to manage load spikes and emergency situations. Our evaluation on a large GPU cluster demonstrates significant reductions in thermal and power throttling events, boosting system efficiency.
    </details>
</div>
