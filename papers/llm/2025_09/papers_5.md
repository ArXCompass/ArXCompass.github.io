# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01566v1">CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As global e-commerce platforms continue to expand, companies are entering new markets where they encounter cold-start challenges due to limited human labels and user behaviors. In this paper, we share our experiences in Coupang to provide a competitive cold-start performance of relevance matching for emerging e-commerce markets. Specifically, we present a Cold-Start Relevance Matching (CSRM) framework, utilizing a multilingual Large Language Model (LLM) to address three challenges: (1) activating cross-lingual transfer learning abilities of LLMs through machine translation tasks; (2) enhancing query understanding and incorporating e-commerce knowledge by retrieval-based query augmentation; (3) mitigating the impact of training label errors through a multi-round self-distillation training strategy. Our experiments demonstrate the effectiveness of CSRM-LLM and the proposed techniques, resulting in successful real-world deployment and significant online gains, with a 45.8% reduction in defect ratio and a 0.866% uplift in session purchase rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01564v1">Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01509v1">Insight-LLM: LLM-enhanced Multi-view Fusion in Insider Threat Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Insider threat detection (ITD) requires analyzing sparse, heterogeneous user behavior. Existing ITD methods predominantly rely on single-view modeling, resulting in limited coverage and missed anomalies. While multi-view learning has shown promise in other domains, its direct application to ITD introduces significant challenges: scalability bottlenecks from independently trained sub-models, semantic misalignment across disparate feature spaces, and view imbalance that causes high-signal modalities to overshadow weaker ones. In this work, we present Insight-LLM, the first modular multi-view fusion framework specifically tailored for insider threat detection. Insight-LLM employs frozen, pre-nes, achieving state-of-the-art detection with low latency and parameter overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01494v1">Benchmarking and Studying the LLM-based Code Review</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Automated Code Review (ACR) is crucial for software quality, yet existing benchmarks often fail to reflect real-world complexities, hindering the evaluation of modern Large Language Models (LLMs). Current benchmarks frequently focus on fine-grained code units, lack complete project context, and use inadequate evaluation metrics. To address these limitations, we introduce SWRBench , a new benchmark comprising 1000 manually verified Pull Requests (PRs) from GitHub, offering PR-centric review with full project context. SWRBench employs an objective LLM-based evaluation method that aligns strongly with human judgment (~90 agreement) by verifying if issues from a structured ground truth are covered in generated reviews. Our systematic evaluation of mainstream ACR tools and LLMs on SWRBench reveals that current systems underperform, and ACR tools are more adept at detecting functional errors. Subsequently, we propose and validate a simple multi-review aggregation strategy that significantly boosts ACR performance, increasing F1 scores by up to 43.67%. Our contributions include the SWRBench benchmark, its objective evaluation method, a comprehensive study of current ACR capabilities, and an effective enhancement approach, offering valuable insights for advancing ACR research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01444v1">Strata-Sword: A Hierarchical Safety Evaluation towards LLMs based on Reasoning Complexity of Jailbreak Instructions</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained widespread recognition for their superior comprehension and have been deployed across numerous domains. Building on Chain-of-Thought (CoT) ideology, Large Reasoning models (LRMs) further exhibit strong reasoning skills, enabling them to infer user intent more accurately and respond appropriately. However, both LLMs and LRMs face the potential safety risks under jailbreak attacks, which raise concerns about their safety capabilities. Current safety evaluation methods often focus on the content dimensions, or simply aggregate different attack methods, lacking consideration of the complexity. In fact, instructions of different complexity can reflect the different safety capabilities of the model: simple instructions can reflect the basic values of the model, while complex instructions can reflect the model's ability to deal with deeper safety risks. Therefore, a comprehensive benchmark needs to be established to evaluate the safety performance of the model in the face of instructions of varying complexity, which can provide a better understanding of the safety boundaries of the LLMs. Thus, this paper first quantifies "Reasoning Complexity" as an evaluable safety dimension and categorizes 15 jailbreak attack methods into three different levels according to the reasoning complexity, establishing a hierarchical Chinese-English jailbreak safety benchmark for systematically evaluating the safety performance of LLMs. Meanwhile, to fully utilize unique language characteristics, we first propose some Chinese jailbreak attack methods, including the Chinese Character Disassembly attack, Lantern Riddle attack, and Acrostic Poem attack. A series of experiments indicate that current LLMs and LRMs show different safety boundaries under different reasoning complexity, which provides a new perspective to develop safer LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01441v1">LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      As the social environment is growing more complex and collaboration is deepening, factors affecting the healthy development of service ecosystem are constantly changing and diverse, making its governance a crucial research issue. Applying the scenario analysis method and conducting scenario rehearsals by constructing an experimental system before managers make decisions, losses caused by wrong decisions can be largely avoided. However, it relies on predefined rules to construct scenarios and faces challenges such as limited information, a large number of influencing factors, and the difficulty of measuring social elements. These challenges limit the quality and efficiency of generating social and uncertain scenarios for the service ecosystem. Therefore, we propose a scenario generator design method, which adaptively coordinates three Large Language Model (LLM) empowered agents that autonomously optimize experimental schemes to construct an experimental system and generate high quality scenarios. Specifically, the Environment Agent (EA) generates social environment including extremes, the Social Agent (SA) generates social collaboration structure, and the Planner Agent (PA) couples task-role relationships and plans task solutions. These agents work in coordination, with the PA adjusting the experimental scheme in real time by perceiving the states of each agent and these generating scenarios. Experiments on the ProgrammableWeb dataset illustrate our method generates more accurate scenarios more efficiently, and innovatively provides an effective way for service ecosystem governance related experimental system construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01412v1">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01395v1">LLMs cannot spot math errors, even when allowed to peek into the solution</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance on math word problems, yet they have been shown to struggle with meta-reasoning tasks such as identifying errors in student solutions. In this work, we investigate the challenge of locating the first error step in stepwise solutions using two error reasoning datasets: VtG and PRM800K. Our experiments show that state-of-the-art LLMs struggle to locate the first error step in student solutions even when given access to the reference solution. To that end, we propose an approach that generates an intermediate corrected student solution, aligning more closely with the original student's solution, which helps improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01393v1">Adaptive Alpha Weighting with PPO: Enhancing Prompt-Based LLM-Generated Alphas in Quant Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      This paper proposes a reinforcement learning framework that employs Proximal Policy Optimization (PPO) to dynamically optimize the weights of multiple large language model (LLM)-generated formulaic alphas for stock trading strategies. Formulaic alphas are mathematically defined trading signals derived from price, volume, sentiment, and other data. Although recent studies have shown that LLMs can generate diverse and effective alphas, a critical challenge lies in how to adaptively integrate them under varying market conditions. To address this gap, we leverage the deepseek-r1-distill-llama-70b model to generate fifty alphas for five major stocks: Apple, HSBC, Pepsi, Toyota, and Tencent, and then use PPO to adjust their weights in real time. Experimental results demonstrate that the PPO-optimized strategy achieves strong returns and high Sharpe ratios across most stocks, outperforming both an equal-weighted alpha portfolio and traditional benchmarks such as the Nikkei 225, S&P 500, and Hang Seng Index. The findings highlight the importance of reinforcement learning in the allocation of alpha weights and show the potential of combining LLM-generated signals with adaptive optimization for robust financial forecasting and trading.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01354v1">DPF-CM: A Data Processing Framework with Privacy-Preserving Vector Databases for Chinese Medical LLMs Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Current open-source training pipelines for Chinese medical language models predominantly emphasize optimizing training methodologies to enhance the performance of large language models (LLMs), yet lack comprehensive exploration into training data processing. To address this gap, we propose DPF-CM, a holistic Data Processing Framework for Chinese Medical LLMs training and deployment. DPF-CM comprises two core modules. The first module is a data processing pipeline tailored for model training. Beyond standard data processing operations, we (1) introduce a chained examples context-learning strategy to generate question-oriented instructions to mitigate the lack of instruction content, and (2) implement an ensemble-based filtering mechanism for preference data curation that averages multiple reward models to suppress noisy samples. The second module focuses on privacy preservation during model deployment. To prevent privacy risks from the inadvertent exposure of training data, we propose a Privacy Preserving Vector Database (PPVD) approach, which involves model memory search, high-risk database construction, secure database construction, and match-and-replace, four key stages to minimize privacy leakage during inference collectively. Experimental results show that DPF-CM significantly improves model accuracy, enabling our trained Chinese medical LLM to achieve state-of-the-art performance among open-source counterparts. Moreover, the framework reduces training data privacy leakage by 27%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01337v1">LLM-Guided Semantic Relational Reasoning for Multimodal Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025 (Main Track, Long Paper)
    </div>
    <details class="paper-abstract">
      Understanding human intents from multimodal signals is critical for analyzing human behaviors and enhancing human-machine interactions in real-world scenarios. However, existing methods exhibit limitations in their modality-level reliance, constraining relational reasoning over fine-grained semantics for complex intent understanding. This paper proposes a novel LLM-Guided Semantic Relational Reasoning (LGSRR) method, which harnesses the expansive knowledge of large language models (LLMs) to establish semantic foundations that boost smaller models' relational reasoning performance. Specifically, an LLM-based strategy is proposed to extract fine-grained semantics as guidance for subsequent reasoning, driven by a shallow-to-deep Chain-of-Thought (CoT) that autonomously uncovers, describes, and ranks semantic cues by their importance without relying on manually defined priors. Besides, we formally model three fundamental types of semantic relations grounded in logical principles and analyze their nuanced interplay to enable more effective relational reasoning. Extensive experiments on multimodal intent and dialogue act recognition tasks demonstrate LGSRR's superiority over state-of-the-art methods, with consistent performance gains across diverse semantic understanding scenarios. The complete data and code are available at https://github.com/thuiar/LGSRR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01314v1">Can Smaller LLMs do better? Unlocking Cross-Domain Potential through Parameter-Efficient Fine-Tuning for Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), being generic task solvers, are versatile. However, despite the vast amount of data they are trained on, there are speculations about their adaptation capabilities to a new domain. Additionally, the simple fine-tuning of the model to incorporate knowledge of a new domain is computationally expensive and time-consuming. This becomes more challenging when the domain in question is also low-resource, and labeled data is unavailable. We leverage parameter-efficient fine-tuning techniques (PEFTs) on high-resource datasets to address these challenges to improve performance on unseen low-resource domains. Throughout our experiments, we evaluate whether intrinsic linguistic commonalities between datasets can be leveraged for efficient domain adaptation. We benchmark six PEFTs with \texttt{Llama-3-8B-Instruct} on 14 training datasets from the Scientific, Medical, Legal, and News domains for a Text Summarization task. Our experiments show that for low-resource domains, inference using Within-Domain Adapters can achieve better performance than Few-Shot as well as a much larger \texttt{Llama-3-70B-Instruct}. Lastly, in the absence of Within-Domain Adapters, we explore the concept of using Cross-Domain Adapters as well as the strategic combinations of adapters to leverage intrinsic language similarities across domains, facilitating better adaptability and performance in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01277v1">Communicative Agents for Slideshow Storytelling Video Generation based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 8 pages, 8 figures, 1 table
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence (AI), the proliferation of AI-generated content (AIGC) tasks has significantly accelerated developments in text-to-video generation. As a result, the field of video production is undergoing a transformative shift. However, conventional text-to-video models are typically constrained by high computational costs. In this study, we propose Video-Generation-Team (VGTeam), a novel slide show video generation system designed to redefine the video creation pipeline through the integration of large language models (LLMs). VGTeam is composed of a suite of communicative agents, each responsible for a distinct aspect of video generation, such as scriptwriting, scene creation, and audio design. These agents operate collaboratively within a chat tower workflow, transforming user-provided textual prompts into coherent, slide-style narrative videos. By emulating the sequential stages of traditional video production, VGTeam achieves remarkable improvements in both efficiency and scalability, while substantially reducing computational overhead. On average, the system generates videos at a cost of only $0.103, with a successful generation rate of 98.4%. Importantly, this framework maintains a high degree of creative fidelity and customization. The implications of VGTeam are far-reaching. It democratizes video production by enabling broader access to high-quality content creation without the need for extensive resources. Furthermore, it highlights the transformative potential of language models in creative domains and positions VGTeam as a pioneering system for next-generation content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01267v1">Iterative In-Context Learning to Enhance LLMs Abstract Reasoning: The Case-Study of Algebraic Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      LLMs face significant challenges in systematic generalization, particularly when dealing with reasoning tasks requiring compositional rules and handling out-of-distribution examples. To address these challenges, we introduce an in-context learning methodology that improves the generalization capabilities of general purpose LLMs. Our approach employs an iterative example selection strategy, which incrementally constructs a tailored set of few-shot examples optimized to enhance model's performance on a given task. As a proof of concept, we apply this methodology to the resolution of algebraic expressions involving non-standard simplification rules, according to which the priority of addition and multiplication is changed. Our findings indicate that LLMs exhibit limited proficiency in these mathematical tasks. We further demonstrate that LLMs reasoning benefits from our iterative shot selection prompting strategy integrated with explicit reasoning instructions. Crucially, our experiments reveal that some LLMs achieve better generalization performances when prompted with simpler few-shot examples rather than complex ones following the test data distribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01229v1">LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 12 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Quantization is a critical technique for accelerating LLM inference by reducing memory footprint and improving computational efficiency. Among various schemes, 4-bit weight and 8-bit activation quantization (W4A8) offers a strong balance between accuracy and performance. However, existing W4A8 GEMM kernels fall short in practice due to inefficient dequantization on CUDA Cores, which cannot keep pace with the high throughput of Tensor Cores. In this paper, we present LiquidGEMM, a hardware-efficient W4A8 GEMM kernel for efficient LLM serving. LiquidGEMM designs two key techniques: LiquidQuant, a hardware-efficient quantization method that enables fast, overflow-safe dequantization using just two arithmetic instructions per four elements; and an implicit fine-grained pipeline that fully overlaps weight loading, dequantization, and MMA across warp groups without software synchronization or redundant memory traffic. Experimental results show that LiquidGEMM achieves up to 2.90x speedup over state-of-the-art W4A8 kernels and up to 4.94x end-to-end system-level speedup. Compared to various quantized GEMM kernels in NVIDIA TensorRT-LLM, LiquidGEMM delivers 1.12-1.63x performance gains, and achieves up to 1.63x system-level speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04492v1">Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 8 pages, 7 figures, 1 table. pre-print version
    </div>
    <details class="paper-abstract">
      Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01211v1">Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      With the proliferation of applications built upon LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern in ensuring system reliability. Once an agent is induced to visit a malicious website, attackers can use it as a springboard to conduct diverse subsequent attacks, which will drastically expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack aiming at inducing MAS to visit malicious websites. We design 11 representative attack variants that encompass domain name tampering (homoglyph deception, character substitution, etc.), link structure camouflage (sub-directory nesting, sub-domain grafting, parameter obfuscation, etc.), and other deceptive techniques tailored to exploit MAS's vulnerabilities in link validation. Through extensive experiments on these crafted attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input formats such as jailbreaking, which inherently carry higher exposure risks. These results underscore the importance of addressing Web fraud attacks in LLM-driven MAS, as their stealthiness and destructiveness pose non-negligible threats to system security and user safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01035v1">We Politely Insist: Your LLM Must Learn the Persian Art of Taarof</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle to navigate culturally specific communication norms, limiting their effectiveness in global contexts. We focus on Persian taarof, a social norm in Iranian interactions, which is a sophisticated system of ritual politeness that emphasizes deference, modesty, and indirectness, yet remains absent from existing cultural benchmarks. We introduce TaarofBench, the first benchmark for evaluating LLM understanding of taarof, comprising 450 role-play scenarios covering 12 common social interaction topics, validated by native speakers. Our evaluation of five frontier LLMs reveals substantial gaps in cultural competence, with accuracy rates 40-48% below native speakers when taarof is culturally appropriate. Performance varies between interaction topics, improves with Persian-language prompts, and exhibits gender-based asymmetries. We also show that responses rated "polite" by standard metrics often violate taarof norms, indicating the limitations of Western politeness frameworks. Through supervised fine-tuning and Direct Preference Optimization, we achieve 21.8% and 42.3% improvement in model alignment with cultural expectations. Our human study with 33 participants (11 native Persian, 11 heritage, and 11 non-Iranian speakers) forms baselines in varying degrees of familiarity with Persian norms. This work lays the foundation for developing diverse and culturally aware LLMs, enabling applications that better navigate complex social interactions.
    </details>
</div>
