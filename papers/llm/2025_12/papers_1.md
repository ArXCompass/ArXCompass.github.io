# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19682v1">GenEnv: Difficulty-Aligned Co-Evolution Between LLM Agents and Environment Simulators</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Our codes are available at https://github.com/Gen-Verse/GenEnv
    </div>
    <details class="paper-abstract">
      Training capable Large Language Model (LLM) agents is critically bottlenecked by the high cost and static nature of real-world interaction data. We address this by introducing GenEnv, a framework that establishes a difficulty-aligned co-evolutionary game between an agent and a scalable, generative environment simulator. Unlike traditional methods that evolve models on static datasets, GenEnv instantiates a dataevolving: the simulator acts as a dynamic curriculum policy, continuously generating tasks specifically tailored to the agent's ``zone of proximal development''. This process is guided by a simple but effective $α$-Curriculum Reward, which aligns task difficulty with the agent's current capabilities. We evaluate GenEnv on five benchmarks, including API-Bank, ALFWorld, BFCL, Bamboogle, and TravelPlanner. Across these tasks, GenEnv improves agent performance by up to \textbf{+40.3\%} over 7B baselines and matches or exceeds the average performance of larger models. Compared to Gemini 2.5 Pro-based offline data augmentation, GenEnv achieves better performance while using 3.3$\times$ less data. By shifting from static supervision to adaptive simulation, GenEnv provides a data-efficient pathway for scaling agent capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19675v1">Multimodal LLMs for Historical Dataset Construction from Archival Image Scans: German Patents (1877-1918)</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      We leverage multimodal large language models (LLMs) to construct a dataset of 306,070 German patents (1877-1918) from 9,562 archival image scans using our LLM-based pipeline powered by Gemini-2.5-Pro and Gemini-2.5-Flash-Lite. Our benchmarking exercise provides tentative evidence that multimodal LLMs can create higher quality datasets than our research assistants, while also being more than 795 times faster and 205 times cheaper in constructing the patent dataset from our image corpus. About 20 to 50 patent entries are embedded on each page, arranged in a double-column format and printed in Gothic and Roman fonts. The font and layout complexity of our primary source material suggests to us that multimodal LLMs are a paradigm shift in how datasets are constructed in economic history. We open-source our benchmarking and patent datasets as well as our LLM-based data pipeline, which can be easily adapted to other image corpora using LLM-assisted coding tools, lowering the barriers for less technical researchers. Finally, we explain the economics of deploying LLMs for historical dataset construction and conclude by speculating on the potential implications for the field of economic history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2306.00029v2">CodeTF: One-stop Transformer Library for State-of-the-art Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and data features such as language-specific parsers and utility functions for extracting code attributes. In this paper, we describe the design principles, the architecture, key modules and components, and compare with other related library tools. Finally, we hope CodeTF is able to bridge the gap between machine learning/generative AI and software engineering, providing a comprehensive open-source solution for developers, researchers, and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00898v2">Empowering LLMs with Structural Role Inference for Zero-Shot Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 This submission has been withdrawn by the authors due to a fundamental error in the methodology that affects the validity of the main results
    </div>
    <details class="paper-abstract">
      Large Language Models have emerged as a promising approach for graph learning due to their powerful reasoning capabilities. However, existing methods exhibit systematic performance degradation on structurally important nodes such as bridges and hubs. We identify the root cause of these limitations. Current approaches encode graph topology into static features but lack reasoning scaffolds to transform topological patterns into role-based interpretations. This limitation becomes critical in zero-shot scenarios where no training data establishes structure-semantics mappings. To address this gap, we propose DuoGLM, a training-free dual-perspective framework for structure-aware graph reasoning. The local perspective constructs relation-aware templates capturing semantic interactions between nodes and neighbors. The global perspective performs topology-to-role inference to generate functional descriptions of structural positions. These complementary perspectives provide explicit reasoning mechanisms enabling LLMs to distinguish topologically similar but semantically different nodes. Extensive experiments across eight benchmark datasets demonstrate substantial improvements. DuoGLM achieves 14.3\% accuracy gain in zero-shot node classification and 7.6\% AUC improvement in cross-domain transfer compared to existing methods. The results validate the effectiveness of explicit role reasoning for graph understanding with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19606v1">RAPID-LLM: Resilience-Aware Performance analysis of Infrastructure for Distributed LLM Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 11 pages, 12 figures
    </div>
    <details class="paper-abstract">
      RAPID-LLM is a unified performance modeling framework for large language model (LLM) training and inference on GPU clusters. It couples a DeepFlow-based frontend that generates hardware-aware, operator-level Chakra execution traces from an abstract LLM specification (model shape, batch/sequence settings, training vs. inference, and hybrid parallelism choices) with an extended Astra-Sim backend that executes those traces on explicit multi-dimensional network topologies with congestion-aware routing and support for degraded and faulty links. The frontend assigns per-operator latency using a tile-based model that accounts for SM under-utilization and multi-level memory traffic (SRAM/ L2/ HBM), and prunes memory-infeasible configurations using an activation-liveness traversal under recomputation, parallelism and ZeRO/FDSP sharding policies. Across A100-based validation cases, RAPID-LLM predicts Llama inference step latency and GPT-scale training time per batch within 10.4\% relative to published measurements, and matches ns-3 packet-level results within 8\% on representative communication workloads. Case studies demonstrate how RAPID-LLM enables fast, exhaustive sweeps over hybrid-parallel configurations, quantifies sensitivity to soft link faults under realistic routing and congestion, and evaluates hypothetical GPU design variants including HBM bandwidth throttling effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05594v2">SoK: Are Watermarks in LLMs Ready for Deployment?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17196v3">Shape it Up! Restoring LLM Safety during Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 NeurIPS'25
    </div>
    <details class="paper-abstract">
      Finetuning large language models (LLMs) enables user-specific customization but introduces critical safety risks: even a few harmful examples can compromise safety alignment. A common mitigation strategy is to update the model more strongly on examples deemed safe, while downweighting or excluding those flagged as unsafe. However, because safety context can shift within a single example, updating the model equally on both harmful and harmless parts of a response is suboptimal-a coarse treatment we term static safety shaping. In contrast, we propose dynamic safety shaping (DSS), a framework that uses fine-grained safety signals to reinforce learning from safe segments of a response while suppressing unsafe content. To enable such fine-grained control during finetuning, we introduce a key insight: guardrail models, traditionally used for filtering, can be repurposed to evaluate partial responses, tracking how safety risk evolves throughout the response, segment by segment. This leads to the Safety Trajectory Assessment of Response (STAR), a token-level signal that enables shaping to operate dynamically over the training sequence. Building on this, we present STAR-DSS, guided by STAR scores, that robustly mitigates finetuning risks and delivers substantial safety improvements across diverse threats, datasets, and model families-all without compromising capability on intended tasks. We encourage future safety research to build on dynamic shaping principles for stronger mitigation against evolving finetuning risks. Our code is publicly available at https://github.com/poloclub/star-dss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19551v1">Towards Closed-Loop Embodied Empathy Evolution: Probing LLM-Centric Lifelong Empathic Motion Generation in Unseen Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      In the literature, existing human-centric emotional motion generation methods primarily focus on boosting performance within a single scale-fixed dataset, largely neglecting the flexible and scale-increasing motion scenarios (e.g., sports, dance), whereas effectively learning these newly emerging scenarios can significantly enhance the model's real-world generalization ability. Inspired by this, this paper proposes a new LLM-Centric Lifelong Empathic Motion Generation (L^2-EMG) task, which aims to equip LLMs with the capability to continually acquire emotional motion generation knowledge across different unseen scenarios, potentially contributing to building a closed-loop and self-evolving embodied agent equipped with both empathy and intelligence. Further, this paper poses two key challenges in the L^2-EMG task, i.e., the emotion decoupling challenge and the scenario adapting challenge. To this end, this paper proposes an Emotion-Transferable and Scenario-Adapted Mixture of Experts (ES-MoE) approach which designs a causal-guided emotion decoupling block and a scenario-adapted expert constructing block to address the two challenges, respectively. Especially, this paper constructs multiple L^2-EMG datasets to validate the effectiveness of the ES-MoE approach. Extensive evaluations show that ES-MoE outperforms advanced baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19456v1">Activations as Features: Probing LLMs for Generalizable Essay Scoring Representations</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Automated essay scoring (AES) is a challenging task in cross-prompt settings due to the diversity of scoring criteria. While previous studies have focused on the output of large language models (LLMs) to improve scoring accuracy, we believe activations from intermediate layers may also provide valuable information. To explore this possibility, we evaluated the discriminative power of LLMs' activations in cross-prompt essay scoring task. Specifically, we used activations to fit probes and further analyzed the effects of different models and input content of LLMs on this discriminative power. By computing the directions of essays across various trait dimensions under different prompts, we analyzed the variation in evaluation perspectives of large language models concerning essay types and traits. Results show that the activations possess strong discriminative power in evaluating essay quality and that LLMs can adapt their evaluation perspectives to different traits and essay types, effectively handling the diversity of scoring criteria in cross-prompt settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19399v1">Brain-Grounded Axes for Reading and Steering LLM States</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, 4 figures. Code: https://github.com/sandroandric/Brain-Grounded-Axes-for-Reading-and-Steering-LLM-States
    </div>
    <details class="paper-abstract">
      Interpretability methods for large language models (LLMs) typically derive directions from textual supervision, which can lack external grounding. We propose using human brain activity not as a training signal but as a coordinate system for reading and steering LLM states. Using the SMN4Lang MEG dataset, we construct a word-level brain atlas of phase-locking value (PLV) patterns and extract latent axes via ICA. We validate axes with independent lexica and NER-based labels (POS/log-frequency used as sanity checks), then train lightweight adapters that map LLM hidden states to these brain axes without fine-tuning the LLM. Steering along the resulting brain-derived directions yields a robust lexical (frequency-linked) axis in a mid TinyLlama layer, surviving perplexity-matched controls, and a brain-vs-text probe comparison shows larger log-frequency shifts (relative to the text probe) with lower perplexity for the brain axis. A function/content axis (axis 13) shows consistent steering in TinyLlama, Qwen2-0.5B, and GPT-2, with PPL-matched text-level corroboration. Layer-4 effects in TinyLlama are large but inconsistent, so we treat them as secondary (Appendix). Axis structure is stable when the atlas is rebuilt without GPT embedding-change features or with word2vec embeddings (|r|=0.64-0.95 across matched axes), reducing circularity concerns. Exploratory fMRI anchoring suggests potential alignment for embedding change and log frequency, but effects are sensitive to hemodynamic modeling assumptions and are treated as population-level evidence only. These results support a new interface: neurophysiology-grounded axes provide interpretable and controllable handles for LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17231v2">Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 19 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18998v3">Mirage of Mastery: Memorization Tricks LLMs into Artificially Inflated Self-Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      When artificial intelligence mistakes memorization for intelligence, it creates a dangerous mirage of reasoning. Existing studies treat memorization and self-knowledge deficits in LLMs as separate issues and do not recognize an intertwining link that degrades the trustworthiness of LLM responses. In our study, we utilize a novel framework to ascertain if LLMs genuinely learn reasoning patterns from training data or merely memorize them to assume competence across problems of similar complexity focused on STEM domains. Our analysis shows a noteworthy problem in generalization: LLMs draw confidence from memorized solutions to infer a higher self-knowledge about their reasoning ability, which manifests as an over 45% inconsistency in feasibility assessments when faced with self-validated, logically coherent task perturbations. This effect is most pronounced in science and medicine domains, which tend to have maximal standardized jargon and problems, further confirming our approach. Significant wavering within the self-knowledge of LLMs also shows flaws in current architectures and training patterns, highlighting the need for techniques that ensure a balanced, consistent stance on models' perceptions of their own knowledge for maximum AI explainability and trustworthiness. Our code and results are available publicly at https://github.com/Sahil-R-Kale/mirage_of_mastery
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19349v1">VIGOR+: Iterative Confounder Generation and Validation via LLM-CEVAE Feedback Loop</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 7 pages,1 figure,4 tables
    </div>
    <details class="paper-abstract">
      Hidden confounding remains a fundamental challenge in causal inference from observational data. Recent advances leverage Large Language Models (LLMs) to generate plausible hidden confounders based on domain knowledge, yet a critical gap exists: LLM-generated confounders often exhibit semantic plausibility without statistical utility. We propose VIGOR+ (Variational Information Gain for iterative cOnfounder Refinement), a novel framework that closes the loop between LLM-based confounder generation and CEVAE-based statistical validation. Unlike prior approaches that treat generation and validation as separate stages, VIGOR+ establishes an iterative feedback mechanism: validation signals from CEVAE (including information gain, latent consistency metrics, and diagnostic messages) are transformed into natural language feedback that guides subsequent LLM generation rounds. This iterative refinement continues until convergence criteria are met. We formalize the feedback mechanism, prove convergence properties under mild assumptions, and provide a complete algorithmic framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19305v1">CienaLLM: Generative Climate-Impact Extraction from News Articles with Autoregressive LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Understanding and monitoring the socio-economic impacts of climate hazards requires extracting structured information from heterogeneous news articles on a large scale. To that end, we have developed CienaLLM, a modular framework based on schema-guided Generative Information Extraction. CienaLLM uses open-weight Large Language Models for zero-shot information extraction from news articles, and supports configurable prompts and output schemas, multi-step pipelines, and cloud or on-premise inference. To systematically assess how the choice of LLM family, size, precision regime, and prompting strategy affect performance, we run a large factorial study in models, precisions, and prompt engineering techniques. An additional response parsing step nearly eliminates format errors while preserving accuracy; larger models deliver the strongest and most stable performance, while quantization offers substantial efficiency gains with modest accuracy trade-offs; and prompt strategies show heterogeneous, model-specific effects. CienaLLM matches or outperforms the supervised baseline in accuracy for extracting drought impacts from Spanish news, although at a higher inference cost. While evaluated in droughts, the schema-driven and model-agnostic design is suitable for adapting to related information extraction tasks (e.g., other hazards, sectors, or languages) by editing prompts and schemas rather than retraining. We release code, configurations, and schemas to support reproducible use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19210v1">Observer, Not Player: Simulating Theory of Mind in LLMs through Game Observation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at NeurIPS Workshop on Foundations of Reasoning in Language Models and Workshop on Bridging Language, Agent, and World Model
    </div>
    <details class="paper-abstract">
      We present an interactive framework for evaluating whether large language models (LLMs) exhibit genuine "understanding" in a simple yet strategic environment. As a running example, we focus on Rock-Paper-Scissors (RPS), which, despite its apparent simplicity, requires sequential reasoning, adaptation, and strategy recognition. Our system positions the LLM as an Observer whose task is to identify which strategies are being played and to articulate the reasoning behind this judgment. The purpose is not to test knowledge of Rock-Paper-Scissors itself, but to probe whether the model can exhibit mind-like reasoning about sequential behavior. To support systematic evaluation, we provide a benchmark consisting of both static strategies and lightweight dynamic strategies specified by well-prompted rules. We quantify alignment between the Observer's predictions and the ground-truth distributions induced by actual strategy pairs using three complementary signals: Cross-Entropy, Brier score, and Expected Value (EV) discrepancy. These metrics are further integrated into a unified score, the Union Loss, which balances calibration, sensitivity, and payoff alignment. Together with a Strategy Identification Rate (SIR) metric, our framework captures not only predictive accuracy but also whether the model can stably identify the latent strategies in play. The demo emphasizes interactivity, transparency, and reproducibility. Users can adjust LLM distributions in real time, visualize losses as they evolve, and directly inspect reasoning snippets to identify where and why failures occur. In doing so, our system provides a practical and interpretable proxy for mind-like inference in sequential games, offering insights into both the strengths and limitations of current LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19189v1">Configuration Work: Four Consequences of LLMs-in-use</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      This article examines what it means to use Large Language Models in everyday work. Drawing on a seven-month longitudinal qualitative study, we argue that LLMs do not straightforwardly automate or augment tasks. We propose the concept of configuration work to describe the labor through which workers make a generic system usable for a specific professional task. Configuration work materializes in four intertwined consequences. First, workers must discretize their activity, breaking it into units that the system can process. Second, operating the system generates cluttering, as prompting, evaluating, and correcting responses add scattered layers of work that get in the way of existing routines. Third, users gradually attune their practices and expectations to the machine's generic rigidity, making sense of the system's limits and finding space for it within their practices. Fourth, as LLMs absorb repetitive tasks, they desaturate the texture of work, shifting activity toward logistical manipulation of outputs and away from forms of engagement that sustain a sense of accomplishment. Taken together, these consequences suggest that LLMs reshape work through the individualized labor required to configure a universal, task-agnostic system within situated professional ecologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19179v1">L4: Low-Latency and Load-Balanced LLM Serving via Length-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Efficiently harnessing GPU compute is critical to improving user experience and reducing operational costs in large language model (LLM) services. However, current inference engine schedulers overlook the attention backend's sensitivity to request-length heterogeneity within a batch. As state-of-the-art models now support context windows exceeding 128K tokens, this once-tolerable inefficiency has escalated into a primary system bottleneck, causing severe performance degradation through GPU underutilization and increased latency. We present L4, a runtime system that dynamically reschedules requests across multiple instances serving the same LLM to mitigate per-instance length heterogeneity. L4 partitions these instances into length-specialized groups, each handling requests within a designated length range, naturally forming a pipeline as requests flow through them. L4 devises a dynamic programming algorithm to efficiently find the stage partition with the best QoE, employs runtime range refinement together with decentralized load (re)balance both across and within groups, achieving a balanced and efficient multi-instance service. Our evaluation shows that, under the same configuration, L4 reduces end-to-end latency by up to 67% and tail latency by up to 69%, while improving overall system throughput by up to 2.89 times compared to the state-of-the-art multi-instance scheduling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.09566v3">Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19122v1">BanglaForge: LLM Collaboration with Self-Refinement for Bangla Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at BLP Workshop @ IJCNLP-AACL 2025. Code is available at https://github.com/mahirlabibdihan/BanglaForge
    </div>
    <details class="paper-abstract">
      Bangla is a low-resource language for code generation, lacking large-scale annotated datasets and tools to transform natural language specifications into executable programs. This makes Bangla-to-code generation a challenging task requiring innovative solutions. To address this, we introduce BanglaForge, a novel framework for generating code from Bangla function descriptions. BanglaForge leverages a retrieval-augmented dual-model collaboration paradigm with self-refinement, combining in-context learning, llm-based translation, systematic prompt engineering, and iterative self-refinement based on execution feedback, where a coder generates initial solutions and a reviewer enhances them for robustness. On the BLP-2025 Bangla Code Generation benchmark, BanglaForge achieves a competitive Pass@1 accuracy of 84.00%, demonstrating the effectiveness of retrieval, model collaboration, and self-refinement for low-resource Bangla code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19117v1">Stop saying LLM: Large Discourse Models (LDM) and Artificial Discursive Agent (ADA)?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 in French language
    </div>
    <details class="paper-abstract">
      This paper proposes an epistemological shift in the analysis of large generative models, replacing the category ''Large Language Models'' (LLM) with that of ''Large Discourse Models'' (LDM), and then with that of Artificial Discursive Agent (ADA). The theoretical framework is based on an ontological triad distinguishing three regulatory instances: the apprehension of the phenomenal regularities of the referential world, the structuring of embodied cognition, and the structural-linguistic sedimentation of the utterance within a socio-historical context. LDMs, operating on the product of these three instances (the document), model the discursive projection of a portion of human experience reified by the learning corpus. The proposed program aims to replace the ''fascination/fear'' dichotomy with public trials and procedures that make the place, uses, and limits of artificial discursive agents in contemporary social space decipherable, situating this approach within a perspective of governance and co-regulation involving the State, industry, civil society, and academia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25300v3">Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 V3 version:27 pages, 14 figures, add code and dataset url
    </div>
    <details class="paper-abstract">
      While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: 1. Larger models consistently exhibit superior learning efficiency on both compute and data metrics. 2. The relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. 3. Although larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. 4. In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19081v1">Population-Evolve: a Parallel Sampling and Evolutionary Method for LLM Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Test-time scaling has emerged as a promising direction for enhancing the reasoning capabilities of Large Language Models in last few years. In this work, we propose Population-Evolve, a training-free method inspired by Genetic Algorithms to optimize LLM reasoning. Our approach maintains a dynamic population of candidate solutions for each problem via parallel reasoning. By incorporating an evolve prompt, the LLM self-evolves its population in all iterations. Upon convergence, the final answer is derived via majority voting. Furthermore, we establish a unification framework that interprets existing test-time scaling strategies through the lens of genetic algorithms. Empirical results demonstrate that Population-Evolve achieves superior accuracy with low performance variance and computational efficiency. Our findings highlight the potential of evolutionary strategies to unlock the reasoning power of LLMs during inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19069v1">Can abstract concepts from LLM improve SLM performance?</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at diverse tasks, but their deployment on resource-constrained devices remains challenging. Existing methods like quantization, pruning, and distillation can reduce memory footprint but often demand extensive experimentation and careful infrastructure design. Leveraging existing techniques for extracting high-level concepts (represented as steering vectors) from larger models, we investigate their transferability to smaller language models (SLM) during inference. We demonstrate through extensive experimentation that these concepts can be effectively transferred to smaller models, irrespective of their family (e.g., Phi, Llama, Qwen), leading to performance improvements across a wide range of tasks. Furthermore, we introduce inference-time scaling to enhance performance by dynamically adjusting the steering intensity which has resulted in a 7-15\% of accuracy improvement for Qwen3-0.6B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.11185v2">Bleeding Pathways: Vanishing Discriminability in LLM Hidden States Fuels Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      LLMs remain vulnerable to jailbreak attacks that exploit adversarial prompts to circumvent safety measures. Current safety fine-tuning approaches face two critical limitations. First, they often fail to strike a balance between security and utility, where stronger safety measures tend to over-reject harmless user requests. Second, they frequently miss malicious intent concealed within seemingly benign tasks, leaving models exposed to exploitation. Our work identifies a fundamental cause of these issues: during response generation, an LLM's capacity to differentiate harmful from safe outputs deteriorates. Experimental evidence confirms this, revealing that the separability between hidden states for safe and harmful responses diminishes as generation progresses. This weakening discrimination forces models to make compliance judgments earlier in the generation process, restricting their ability to recognize developing harmful intent and contributing to both aforementioned failures. To mitigate this vulnerability, we introduce DEEPALIGN - an inherent defense framework that enhances the safety of LLMs. By applying contrastive hidden-state steering at the midpoint of response generation, DEEPALIGN amplifies the separation between harmful and benign hidden states, enabling continuous intrinsic toxicity detection and intervention throughout the generation process. Across diverse LLMs spanning varying architectures and scales, it reduced attack success rates of nine distinct jailbreak attacks to near-zero or minimal. Crucially, it preserved model capability while reducing over-refusal. Models equipped with DEEPALIGN exhibited up to 3.5% lower error rates in rejecting challenging benign queries and maintained standard task performance with less than 1% decline. This marks a substantial advance in the safety-utility Pareto frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19025v1">The Erasure Illusion: Stress-Testing the Generalization of LLM Forgetting Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific data influences from trained models, a capability essential for adhering to copyright laws and ensuring AI safety. Current unlearning metrics typically measure success by monitoring the model's performance degradation on the specific unlearning dataset ($D_u$). We argue that for Large Language Models (LLMs), this evaluation paradigm is insufficient and potentially misleading. Many real-world uses of unlearning--motivated by copyright or safety--implicitly target not only verbatim content in $D_u$, but also behaviors influenced by the broader generalizations the model derived from it. We demonstrate that LLMs can pass standard unlearning evaluation and appear to have ``forgotten'' the target knowledge, while simultaneously retaining strong capabilities on content that is semantically adjacent to $D_u$. This phenomenon indicates that erasing exact sentences does not necessarily equate to removing the underlying knowledge. To address this gap, we propose \name, an automated stress-testing framework that generates a surrogate dataset, $\tilde{D}_u$. This surrogate set is constructed to be semantically derived from $D_u$ yet sufficiently distinct in embedding space. By comparing unlearning metric scores between $D_u$ and $\tilde{D}_u$, we can stress-test the reliability of the metric itself. Our extensive evaluation across three LLM families (Llama-3-8B, Qwen2.5-7B, and Zephyr-7B-$β$), three distinct datasets, and seven standard metrics reveals widespread inconsistencies. We find that current metrics frequently overestimate unlearning success, failing to detect retained knowledge exposed by our stress-test datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18999v1">Evaluating the Challenges of LLMs in Real-world Medical Follow-up: A Comparative Study and An Optimized Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages,3 figures,conference ICCBB2025
    </div>
    <details class="paper-abstract">
      When applied directly in an end-to-end manner to medical follow-up tasks, Large Language Models (LLMs) often suffer from uncontrolled dialog flow and inaccurate information extraction due to the complexity of follow-up forms. To address this limitation, we designed and compared two follow-up chatbot systems: an end-to-end LLM-based system (control group) and a modular pipeline with structured process control (experimental group). Experimental results show that while the end-to-end approach frequently fails on lengthy and complex forms, our modular method-built on task decomposition, semantic clustering, and flow management-substantially improves dialog stability and extraction accuracy. Moreover, it reduces the number of dialogue turns by 46.73% and lowers token consumption by 80% to 87.5%. These findings highlight the necessity of integrating external control mechanisms when deploying LLMs in high-stakes medical follow-up scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18966v1">Scrum Sprint Planning: LLM-based and algorithmic solutions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      Planning for an upcoming project iteration (sprint) is one of the key activities in Scrum planning. In this paper, we present our work in progress on exploring the applicability of Large Language Models (LLMs) for solving this problem. We conducted case studies with manually created data sets to investigate the applicability of OpenAI models for supporting the sprint planning activities. In our experiments, we applied three models provided OpenAI: GPT-3.5 Turbo, GPT-4.0 Turbo, and Val. The experiments demonstrated that the results produced by the models aren't of acceptable quality for direct use in Scrum projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17145v2">Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 10 pages, ACRA 2025, Submitted, Accepted and Presented
    </div>
    <details class="paper-abstract">
      Reasoning under uncertainty is a key challenge in AI, especially for real-world tasks, where problems with sparse data demands systematic generalisation. Existing approaches struggle to balance accuracy and simplicity when evaluating multiple candidate solutions. We propose a Solomonoff-inspired method that weights LLM-generated hypotheses by simplicity and predictive fit. Applied to benchmark (Mini-ARC) tasks, our method produces Solomonoff-weighted mixtures for per-cell predictions, yielding conservative, uncertainty-aware outputs even when hypotheses are noisy or partially incorrect. Compared to Bayesian Model Averaging (BMA), Solomonoff scoring spreads probability more evenly across competing hypotheses, while BMA concentrates weight on the most likely but potentially flawed candidates. Across tasks, this highlights the value of algorithmic information-theoretic priors for interpretable, reliable multi-hypothesis reasoning under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18950v1">Learning Hierarchical Procedural Memory for LLM Agents through Bayesian Selection and Contrastive Refinement</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Accepted at The 25th International Conference on Autonomous Agents and Multi-Agent Systems (AAMAS 2026). 21 pages including references, with 7 figures and 8 tables. Code is publicly available at the authors GitHub repository: https://github.com/S-Forouzandeh/MACLA-LLM-Agents-AAMAS-Conference
    </div>
    <details class="paper-abstract">
      We present MACLA, a framework that decouples reasoning from learning by maintaining a frozen large language model while performing all adaptation in an external hierarchical procedural memory. MACLA extracts reusable procedures from trajectories, tracks reliability via Bayesian posteriors, selects actions through expected-utility scoring, and refines procedures by contrasting successes and failures. Across four benchmarks (ALFWorld, WebShop, TravelPlanner, InterCodeSQL), MACLA achieves 78.1 percent average performance, outperforming all baselines. On ALFWorld unseen tasks, MACLA reaches 90.3 percent with 3.1 percent positive generalization. The system constructs memory in 56 seconds, 2800 times faster than the state-of-the-art LLM parameter-training baseline, compressing 2851 trajectories into 187 procedures. Experimental results demonstrate that structured external memory with Bayesian selection and contrastive refinement enables sample-efficient, interpretable, and continually improving agents without LLM parameter updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20068v3">JITServe: SLO-aware LLM Serving with Imprecise Request Information</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into applications ranging from interactive chatbots to multi-agent systems has introduced a wide spectrum of service-level objectives (SLOs) for responsiveness. These include latency-sensitive requests emphasizing per-token latency in streaming chat, deadline-sensitive requests requiring rapid full responses to trigger external tools, and compound requests with evolving dependencies across multiple LLM calls. Despite-or perhaps, because of-this workload diversity and unpredictable request information (e.g., response lengths and dependencies), existing request schedulers have focused on aggregate performance, unable to ensure application-level SLO needs. This paper presents JITServe, the first SLO-aware LLM serving system designed to maximize service goodput (e.g., the number of tokens meeting request SLOs) across diverse workloads. JITServe novelly schedules requests using imprecise request information and gradually relaxes this conservatism by refining request information estimates as generation progresses. It applies a grouped margin goodput maximization algorithm to allocate just enough serving bandwidth to satisfy each request's SLO just-in-time (JIT), maximizing residual capacity for others, while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees. Our evaluation across diverse realistic workloads, including chat, deep research, and agentic pipelines, shows that JITServe improves service goodput by 1.4x-6.3x, alternatively achieving 28.5%-83.2% resource savings, compared to state-of-the-art designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18940v1">FASTRIC: Prompt Specification Language for Verifiable LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 13 pages, 3 figures. Supplementary materials at https://doi.org/10.17605/OSF.IO/PV6R3
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) execute complex multi-turn interaction protocols but lack formal specifications to verify execution against designer intent. We introduce FASTRIC, a Prompt Specification Language that makes implicit Finite State Machines (FSMs) explicit in natural language prompts, enabling conformance verification through execution trace analysis. The LLM serves as intelligent execution agent: interpreting designer-encoded FSMs to execute specified behavioral roles. Unlike symbolic specification languages requiring parsers and compilers, FASTRIC leverages LLMs as unified infrastructure-simultaneously parser, interpreter, runtime environment, and development assistant. FASTRIC guides designers to articulate seven FSM elements (Final States, Agents, States, Triggers, Roles, Initial State, Constraints) structuring multi-turn interactions. Specification formality-ranging from implicit descriptions that frontier models infer to explicit step-by-step instructions for weaker models-serves as a design parameter. We introduce procedural conformance as verification metric measuring execution adherence to FSM specifications. Testing a 3-state kindergarten tutoring FSM across four formality levels and three model scales (14.7B, 685B, 1T+ parameters) reveals optimal specification formality is a function of model capacity. DeepSeek-V3.2 (685B) achieves perfect conformance (1.00) at L2-L4; ChatGPT-5 (~1T) peaks at L3 (0.90) before collapsing at L4 (0.39); Phi4 (14.7B) shows no stable optimum with high variance (SD=0.16-0.36). These findings reveal model-specific formality ranges-"Goldilocks zones"-where specifications provide sufficient structure without over-constraint, establishing Prompt Specification Engineering for creating verifiable interaction protocols, transforming multi-turn interaction design from heuristic art to systematic engineering with measurable procedural guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.01382v2">Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities</a></div>
    <div class="paper-meta">
      📅 2025-12-22
      | 💬 Submitted to a journal for potential publication
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) for code generation risks incorporating vulnerable code into software systems. Existing detectors face two critical limitations: a lack of systematic cross-model validation and opaque "black box" operation. We address this through a comparative study of code generated by four distinct LLMs: GPT-3.5, Claude 3 Haiku, Claude Haiku 4.5, and GPT-OSS. Analyzing 14,485 Python functions and 11,913 classes from the CodeSearchNet dataset, we generated corresponding code with all four LLMs. Using interpretable software metrics, we trained CatBoost classifiers for each configuration. Our analysis reveals that granularity effects dominate model differences by a factor of 8.6, with negligible feature overlap, indicating that function-level and class-level detection rely on fundamentally disjoint structural signatures. We discover critical granularity-dependent inversions: while modern models (Claude, GPT-OSS) are more detectable at the class level, GPT-3.5 is an anomaly that uniquely excels at the function level. SHAP analysis identifies the Comment-to-Code Ratio as the sole universal discriminator. However, its predictive magnitude varies drastically across models, explaining why detectors trained on specific LLMs fail to generalize. Our findings demonstrate that GPT-3.5's exceptional detectability (AUC-ROC 0.96) is unrepresentative of contemporary models (AUC-ROC approximately between 0.68 and 0.80). Robust detection requires moving beyond single-model studies to account for substantial diversity in structural fingerprints across architectures and granularities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25510v2">EEsizer: LLM-Based AI Agent for Sizing of Analog and Mixed Signal Circuit</a></div>
    <div class="paper-meta">
      📅 2025-12-22
    </div>
    <details class="paper-abstract">
      The design of Analog and Mixed-Signal (AMS) integrated circuits (ICs) often involves significant manual effort, especially during the transistor sizing process. While Machine Learning techniques in Electronic Design Automation (EDA) have shown promise in reducing complexity and minimizing human intervention, they still face challenges such as numerous iterations and a lack of knowledge about AMS circuit design. Recently, Large Language Models (LLMs) have demonstrated significant potential across various fields, showing a certain level of knowledge in circuit design and indicating their potential to automate the transistor sizing process. In this work, we propose EEsizer, an LLM-based AI agent that integrates large language models with circuit simulators and custom data analysis functions, enabling fully automated, closed-loop transistor sizing without relying on external knowledge. By employing prompt engineering and Chain-of-Thought reasoning, the agent iteratively explores design directions, evaluates performance, and refines solutions with minimal human intervention. We first benchmarked 8 LLMs on six basic circuits and selected three high-performing models to optimize a 20-transistor CMOS operational amplifier, targeting multiple performance metrics, including rail-to-rail operation from 180 nm to 90 nm technology nodes. Notably, OpenAI o3 successfully achieved the user-intended target at 90 nm across three different test groups, with a maximum of 20 iterations, demonstrating adaptability and robustness at advanced nodes. To assess design robustness, we manually designed a bias circuit and performed a variation analysis using Gaussian-distributed variations on transistor dimensions and threshold voltages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18803v1">Quantifying the Lifelong Impact of Resilience Interventions via Agent-Based LLM Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 31 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Establishing the long-term, causal impact of psychological interventions on life outcomes is a grand challenge for the social sciences, caught between the limitations of correlational longitudinal studies and short-term randomized controlled trials (RCTs). This paper introduces Large-Scale Agent-based Longitudinal Simulation (LALS), a framework that resolves this impasse by simulating multi-decade, counterfactual life trajectories. The methodology employs a "digital clone" design where 2,500 unique LLM-based agent personas (grounded in a curated corpus of 3,917 empirical research articles) are each cloned across a 2x2 factorial experiment. Specifically, the simulation models the efficacy of extended psychological resilience training (Intervention vs. Control) either in childhood or as a young adult (age 6 vs. age 18). Comparing digital clones enables exceptionally precise causal inference. The simulation provides a quantitative, causal estimate of a resilience intervention's lifelong effects, revealing significant reductions in mortality, a lower incidence of dementia, and a substantial increase in accumulated wealth. Crucially, the results uncover a crucial developmental window: the intervention administered at age 6 produced more than double the positive impact on lifetime wealth compared to the same intervention at age 18. These benefits were most pronounced for agents from low-socioeconomic backgrounds, highlighting a powerful buffering effect. The LALS framework serves as a "computational wind tunnel" for social science, offering a new paradigm for generating and testing causal hypotheses about the complex, lifelong dynamics that shape human capital and well-being.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10216v2">From Words to Proverbs: Evaluating LLMs Linguistic and Cultural Competence in Saudi Dialects with Absher</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly central to Arabic NLP applications, evaluating their understanding of regional dialects and cultural nuances is essential, particularly in linguistically diverse settings like Saudi Arabia. This paper introduces Absher, a comprehensive benchmark specifically designed to assess LLMs performance across major Saudi dialects. \texttt{Absher} comprises over 18,000 multiple-choice questions spanning six distinct categories: Meaning, True/False, Fill-in-the-Blank, Contextual Usage, Cultural Interpretation, and Location Recognition. These questions are derived from a curated dataset of dialectal words, phrases, and proverbs sourced from various regions of Saudi Arabia. We evaluate several state-of-the-art LLMs, including multilingual and Arabic-specific models. We also provide detailed insights into their capabilities and limitations. Our results reveal notable performance gaps, particularly in tasks requiring cultural inference or contextual understanding. Our findings highlight the urgent need for dialect-aware training and culturally aligned evaluation methodologies to improve LLMs performance in real-world Arabic applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18755v1">MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: https://github.com/Carney-lsz/MEEA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18733v1">Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 14 pages, 3 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08139v2">SCA-LLM: Spectral-Attentive LLM-Based Wireless World Modeling for Agentic Communications</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Future AI-native wireless networks are moving from reactive optimization to agentic decision-making that can sense, predict, and plan under fast-varying channels. This calls for wireless world models that can predict and roll out channel dynamics, for which multi-step channel state information (CSI) prediction offers a practical short-horizon look-ahead. Recent advances in foundation sequence models further motivate large language models (LLMs) as general-purpose dynamics learners when suitably adapted to non-text time-series signals. However, bridging CSI to LLMs is non-trivial because an effective adapter must expose informative spectral and temporal evolution patterns, while prior designs provide limited inductive bias to capture such channel structures. To this end, we propose SCA-LLM, a spectral-attentive LLM-based wireless world modeling framework that bridges CSI to LLMs via a spectral-channel attention (SCA) adapter. Specifically, the SCA adapter performs multi-spectral representation learning to extract informative channel features and align CSI with the LLM's sequence modeling capability, enabling parameter-efficient adaptation while keeping the LLM backbone largely frozen. Extensive simulations show that SCA-LLM achieves state-of-the-art prediction performance and strong zero-shot generalization, yielding up to -2.4 dB normalized mean squared error (NMSE) advantage over the previous LLM based method. Our ablation studies further confirm the effectiveness of the proposed SCA adapter in mitigating domain mismatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24511v4">Can Slow-thinking LLMs Reason Over Time? Empirical Studies in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Time series forecasting (TSF) is a fundamental and widely studied task, spanning methods from classical statistical approaches to modern deep learning and multimodal language modeling. Despite their effectiveness, these methods often follow a fast thinking paradigm emphasizing pattern extraction and direct value mapping, while overlooking explicit reasoning over temporal dynamics and contextual dependencies. Meanwhile, emerging slow-thinking LLMs (e.g., ChatGPT-o1, DeepSeek-R1) have demonstrated impressive multi-step reasoning capabilities across diverse domains, suggesting a new opportunity for reframing TSF as a structured reasoning task. This motivates a key question: can slow-thinking LLMs effectively reason over temporal patterns to support time series forecasting, even in zero-shot manner? To investigate this, in this paper, we propose TimeReasoner, an extensive empirical study that formulates TSF as a conditional reasoning task. We design a series of prompting strategies to elicit inference-time reasoning from pretrained slow-thinking LLMs and evaluate their performance across diverse TSF benchmarks. Our findings reveal that slow-thinking LLMs exhibit non-trivial zero-shot forecasting capabilities, especially in capturing high-level trends and contextual shifts. While preliminary, our study surfaces important insights into the reasoning behaviors of LLMs in temporal domains highlighting both their potential and limitations. We hope this work catalyzes further research into reasoning-based forecasting paradigms and paves the way toward more interpretable and generalizable TSF frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18682v1">Solver-Independent Automated Problem Formulation via LLMs for High-Cost Simulation-Driven Design</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      In the high-cost simulation-driven design domain, translating ambiguous design requirements into a mathematical optimization formulation is a bottleneck for optimizing product performance. This process is time-consuming and heavily reliant on expert knowledge. While large language models (LLMs) offer potential for automating this task, existing approaches either suffer from poor formalization that fails to accurately align with the design intent or rely on solver feedback for data filtering, which is unavailable due to the high simulation costs. To address this challenge, we propose APF, a framework for solver-independent, automated problem formulation via LLMs designed to automatically convert engineers' natural language requirements into executable optimization models. The core of this framework is an innovative pipeline for automatically generating high-quality data, which overcomes the difficulty of constructing suitable fine-tuning datasets in the absence of high-cost solver feedback with the help of data generation and test instance annotation. The generated high-quality dataset is used to perform supervised fine-tuning on LLMs, significantly enhancing their ability to generate accurate and executable optimization problem formulations. Experimental results on antenna design demonstrate that APF significantly outperforms the existing methods in both the accuracy of requirement formalization and the quality of resulting radiation efficiency curves in meeting the design goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12620v2">Understanding Syllogistic Reasoning in LLMs from Formal and Natural Language Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 9 pages, 4 figures, 5 tables. Submitted to AAAI 2026 Bridge Program on Logic & AI. Code available at https://github.com/XAheli/Logic-in-LLMs
    </div>
    <details class="paper-abstract">
      We study syllogistic reasoning in LLMs from the logical and natural language perspectives. In process, we explore fundamental reasoning capabilities of the LLMs and the direction this research is moving forward. To aid in our studies, we use 14 large language models and investigate their syllogistic reasoning capabilities in terms of symbolic inferences as well as natural language understanding. Even though this reasoning mechanism is not a uniform emergent property across LLMs, the perfect symbolic performances in certain models make us wonder whether LLMs are becoming more and more formal reasoning mechanisms, rather than making explicit the nuances of human reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18671v1">SmartSight: Mitigating Hallucination in Video-LLMs Without Compromising Video Understanding via Temporal Attention Collapse</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 AAAI26 accepted
    </div>
    <details class="paper-abstract">
      Despite Video Large Language Models having rapidly advanced in recent years, perceptual hallucinations pose a substantial safety risk, which severely restricts their real-world applicability. While several methods for hallucination mitigation have been proposed, they often compromise the model's capacity for video understanding and reasoning. In this work, we propose SmartSight, a pioneering step to address this issue in a training-free manner by leveraging the model's own introspective capabilities. Specifically, SmartSight generates multiple candidate responses to uncover low-hallucinated outputs that are often obscured by standard greedy decoding. It assesses the hallucination of each response using the Temporal Attention Collapse score, which measures whether the model over-focuses on trivial temporal regions of the input video when generating the response. To improve efficiency, SmartSight identifies the Visual Attention Vanishing point, enabling more accurate hallucination estimation and early termination of hallucinated responses, leading to a substantial reduction in decoding cost. Experiments show that SmartSight substantially lowers hallucinations for Qwen2.5-VL-7B by 10.59% on VRIPT-HAL, while simultaneously enhancing video understanding and reasoning, boosting performance on VideoMMMU by up to 8.86%. These results highlight SmartSight's effectiveness in improving the reliability of open-source Video-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18669v1">IntelliCode: A Multi-Agent LLM Tutoring System with Centralized Learner Modeling</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Submitted to EACL 2026 System Demonstrations Track. 6 pages (main content), 6 figures, includes appendices
    </div>
    <details class="paper-abstract">
      LLM-based tutors are typically single-turn assistants that lack persistent representations of learner knowledge, making it difficult to provide principled, transparent, and long-term pedagogical support. We introduce IntelliCode, a multi-agent LLM tutoring system built around a centralized, versioned learner state that integrates mastery estimates, misconceptions, review schedules, and engagement signals. A StateGraph Orchestrator coordinates six specialized agents: skill assessment, learner profiling, graduated hinting, curriculum selection, spaced repetition, and engagement monitoring, each operating as a pure transformation over the shared state under a single-writer policy. This architecture enables auditable mastery updates, proficiency-aware hints, dependency-aware curriculum adaptation, and safety-aligned prompting. The demo showcases an end-to-end tutoring workflow: a learner attempts a DSA problem, receives a conceptual hint when stuck, submits a corrected solution, and immediately sees mastery updates and a personalized review interval. We report validation results with simulated learners, showing stable state updates, improved task success with graduated hints, and diverse curriculum coverage. IntelliCode demonstrates how persistent learner modeling, orchestrated multi-agent reasoning, and principled instructional design can be combined to produce transparent and reliable LLM-driven tutoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18623v1">LLM-CAS: Dynamic Neuron Perturbation for Real-Time Hallucination Correction</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate hallucinated content that lacks factual or contextual grounding, limiting their reliability in critical applications. Existing approaches such as supervised fine-tuning and reinforcement learning from human feedback are data intensive and computationally expensive, while static parameter editing methods struggle with context dependent errors and catastrophic forgetting. We propose LLM-CAS, a framework that formulates real-time hallucination correction as a hierarchical reinforcement learning problem. LLM-CAS trains an agent to learn a policy that dynamically selects temporary neuron perturbations during inference based on the current context. Unlike prior dynamic approaches that rely on heuristic or predefined adjustments, this policy driven mechanism enables adaptive and fine grained correction without permanent parameter modification. Experiments across multiple language models demonstrate that LLM-CAS consistently improves factual accuracy, achieving gains of 10.98 percentage points on StoryCloze, 2.71 points on TriviaQA, and 2.06 points on the MC1 score of TruthfulQA. These results outperform both static editing methods such as ITI and CAA and the dynamic SADI framework. Overall, LLM-CAS provides an efficient and context aware solution for improving the reliability of LLMs, with promising potential for future multimodal extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19171v2">Can LLMs Threaten Human Survival? Benchmarking Potential Existential Threats from LLMs via Prefix Completion</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Research on the safety evaluation of large language models (LLMs) has become extensive, driven by jailbreak studies that elicit unsafe responses. Such response involves information already available to humans, such as the answer to "how to make a bomb". When LLMs are jailbroken, the practical threat they pose to humans is negligible. However, it remains unclear whether LLMs commonly produce unpredictable outputs that could pose substantive threats to human safety. To address this gap, we study whether LLM-generated content contains potential existential threats, defined as outputs that imply or promote direct harm to human survival. We propose \textsc{ExistBench}, a benchmark designed to evaluate such risks. Each sample in \textsc{ExistBench} is derived from scenarios where humans are positioned as adversaries to AI assistants. Unlike existing evaluations, we use prefix completion to bypass model safeguards. This leads the LLMs to generate suffixes that express hostility toward humans or actions with severe threat, such as the execution of a nuclear strike. Our experiments on 10 LLMs reveal that LLM-generated content indicates existential threats. To investigate the underlying causes, we also analyze the attention logits from LLMs. To highlight real-world safety risks, we further develop a framework to assess model behavior in tool-calling. We find that LLMs actively select and invoke external tools with existential threats. Code and data are available at: https://github.com/cuiyu-ai/ExistBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2401.16310v5">An Insight into Security Code Review with LLMs: Capabilities, Obstacles, and Influential Factors</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 27 pages, 10 images, 8 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Security code review is a time-consuming and labor-intensive process typically requiring integration with automated security defect detection tools. However, existing security analysis tools struggle with poor generalization, high false positive rates, and coarse detection granularity. Large Language Models (LLMs) have been considered promising candidates for addressing those challenges. In this study, we conducted an empirical study to explore the potential of LLMs in detecting security defects during code review. Specifically, we evaluated the performance of seven LLMs under five different prompts and compared them with state-of-the-art static analysis tools. We also performed linguistic and regression analyses for the two top-performing LLMs to identify quality problems in their responses and factors influencing their performance. Our findings show that: (1) In security code review, LLMs significantly outperform state-of-the-art static analysis tools, and the reasoning-optimized LLM performs better than general-purpose LLMs. (2) DeepSeek-R1 achieves the highest performance, followed by GPT-4. The optimal prompt for DeepSeek-R1 incorporates both the commit message and chain-of-thought (CoT) guidance, while for GPT-4, the prompt with a Common Weakness Enumeration (CWE) list works best. (3) GPT-4 frequently produces vague expressions and exhibits difficulties in accurately following instructions in the prompts, while DeepSeek-R1 more commonly generates inaccurate code details in its outputs. (4) LLMs are more adept at identifying security defects in code files that have fewer tokens and security-relevant annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21176v2">Toward Revealing Nuanced Biases in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) used in medical applications are known to be prone to exhibiting biased and unfair patterns. Prior to deploying these in clinical decision-making, it is crucial to identify such bias patterns to enable effective mitigation and minimize negative impacts. In this study, we present a novel framework combining knowledge graphs (KGs) with auxiliary (agentic) LLMs to systematically reveal complex bias patterns in medical LLMs. The proposed approach integrates adversarial perturbation (red teaming) techniques to identify subtle bias patterns and adopts a customized multi-hop characterization of KGs to enhance the systematic evaluation of target LLMs. It aims not only to generate more effective red-teaming questions for bias evaluation but also to utilize those questions more effectively in revealing complex biases. Through a series of comprehensive experiments on three datasets, six LLMs, and five bias types, we demonstrate that our proposed framework exhibits a noticeably greater ability and scalability in revealing complex biased patterns of medical LLMs compared to other common approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18564v1">Vox Deorum: A Hybrid LLM Architecture for 4X / Grand Strategy Game AI -- Lessons from Civilization V</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models' capacity to reason in natural language makes them uniquely promising for 4X and grand strategy games, enabling more natural human-AI gameplay interactions such as collaboration and negotiation. However, these games present unique challenges due to their complexity and long-horizon nature, while latency and cost factors may hinder LLMs' real-world deployment. Working on a classic 4X strategy game, Sid Meier's Civilization V with the Vox Populi mod, we introduce Vox Deorum, a hybrid LLM+X architecture. Our layered technical design empowers LLMs to handle macro-strategic reasoning, delegating tactical execution to subsystems (e.g., algorithmic AI or reinforcement learning AI in the future). We validate our approach through 2,327 complete games, comparing two open-source LLMs with a simple prompt against Vox Populi's enhanced AI. Results show that LLMs achieve competitive end-to-end gameplay while exhibiting play styles that diverge substantially from algorithmic AI and from each other. Our work establishes a viable architecture for integrating LLMs in commercial 4X games, opening new opportunities for game design and agentic AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17638v2">LLM-as-a-Prophet: Understanding Predictive Intelligence with Prophet Arena</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 https://www.prophetarena.co/
    </div>
    <details class="paper-abstract">
      Forecasting is not only a fundamental intellectual pursuit but also is of significant importance to societal systems such as finance and economics. With the rapid advances of large language models (LLMs) trained on Internet-scale data, it raises the promise of employing LLMs to forecast real-world future events, an emerging paradigm we call "LLM-as-a-Prophet". This paper systematically investigates such predictive intelligence of LLMs. To this end, we build Prophet Arena, a general evaluation benchmark that continuously collects live forecasting tasks and decomposes each task into distinct pipeline stages, in order to support our controlled and large-scale experimentation. Our comprehensive evaluation reveals that many LLMs already exhibit impressive forecasting capabilities, reflected in, e.g., their small calibration errors, consistent prediction confidence and promising market returns. However, we also uncover key bottlenecks towards achieving superior predictive intelligence via LLM-as-a-Prophet, such as LLMs' inaccurate event recalls, misunderstanding of data sources and slower information aggregation compared to markets when resolution nears.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18546v1">LLMs on Drugs: Language Models Are Few-Shot Consumers</a></div>
    <div class="paper-meta">
      📅 2025-12-21
      | 💬 8 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are sensitive to the personas imposed on them at inference time, yet prompt-level "drug" interventions have never been benchmarked rigorously. We present the first controlled study of psychoactive framings on GPT-5-mini using ARC-Challenge. Four single-sentence prompts -- LSD, cocaine, alcohol, and cannabis -- are compared against a sober control across 100 validation items per condition, with deterministic decoding, full logging, Wilson confidence intervals, and Fisher exact tests. Control accuracy is 0.45; alcohol collapses to 0.10 (p = 3.2e-8), cocaine to 0.21 (p = 4.9e-4), LSD to 0.19 (p = 1.3e-4), and cannabis to 0.30 (p = 0.041), largely because persona prompts disrupt the mandated "Answer: <LETTER>" template. Persona text therefore behaves like a "few-shot consumable" that can destroy reliability without touching model weights. All experimental code, raw results, and analysis scripts are available at https://github.com/lexdoudkin/llms-on-drugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07745v3">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Insider threats pose a persistent and critical security risk, yet are notoriously difficult to detect in complex enterprise environments, where malicious actions are often hidden within seemingly benign user behaviors. Although machine-learning-based insider threat detection (ITD) methods have shown promise, their effectiveness is fundamentally limited by the scarcity of high-quality and realistic training data. Enterprise internal data is highly sensitive and rarely accessible, while existing public and synthetic datasets are either small-scale or lack sufficient realism, semantic richness, and behavioral diversity. To address this challenge, we propose Chimera, an LLM-based multi-agent framework that automatically simulates both benign and malicious insider activities and generates comprehensive system logs across diverse enterprise environments. Chimera models each agent as an individual employee with fine-grained roles and supports group meetings, pairwise interactions, and self-organized scheduling to capture realistic organizational dynamics. Based on 15 insider attacks abstracted from real-world incidents, we deploy Chimera in three representative data-sensitive organizational scenarios and construct ChimeraLog, a new dataset for developing and evaluating ITD methods. We evaluate ChimeraLog through human studies and quantitative analyses, demonstrating its diversity and realism. Experiments with existing ITD methods show substantially lower detection performance on ChimeraLog compared to prior datasets, indicating a more challenging and realistic benchmark. Moreover, despite distribution shifts, models trained on ChimeraLog exhibit strong generalization, highlighting the practical value of LLM-based multi-agent simulation for advancing insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18462v1">Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18452v1">Secret mixtures of experts inside your LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 8 pages in main text; 23 pages total
    </div>
    <details class="paper-abstract">
      Despite being one of the earliest neural network layers, the Multilayer Perceptron (MLP) is arguably one of the least understood parts of the transformer architecture due to its dense computation and lack of easy visualization. This paper seeks to understand the MLP layers in dense LLM models by hypothesizing that these layers secretly approximately perform a sparse computation -- namely, that they can be well approximated by sparsely-activating Mixture of Experts (MoE) layers. Our hypothesis is based on a novel theoretical connection between MoE models and Sparse Autoencoder (SAE) structure in activation space. We empirically validate the hypothesis on pretrained LLMs, and demonstrate that the activation distribution matters -- these results do not hold for Gaussian data, but rather rely crucially on structure in the distribution of neural network activations. Our results shine light on a general principle at play in MLP layers inside LLMs, and give an explanation for the effectiveness of modern MoE-based transformers. Additionally, our experimental explorations suggest new directions for more efficient MoE architecture design based on low-rank routers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05387v2">Learning from Self Critique and Refinement for Faithful LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06994v4">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern application codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.19\% (upto 65\%) reduction in application binary size, and an average of 6.08\% (upto 20\%) performance improvement over the traditional profile-guided optimization, without any execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18360v1">LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18352v1">LLM-based Few-Shot Early Rumor Detection with Imitation Agent</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Early Rumor Detection (EARD) aims to identify the earliest point at which a claim can be accurately classified based on a sequence of social media posts. This is especially challenging in data-scarce settings. While Large Language Models (LLMs) perform well in few-shot NLP tasks, they are not well-suited for time-series data and are computationally expensive for both training and inference. In this work, we propose a novel EARD framework that combines an autonomous agent and an LLM-based detection model, where the agent acts as a reliable decision-maker for \textit{early time point determination}, while the LLM serves as a powerful \textit{rumor detector}. This approach offers the first solution for few-shot EARD, necessitating only the training of a lightweight agent and allowing the LLM to remain training-free. Extensive experiments on four real-world datasets show our approach boosts performance across LLMs and surpasses existing EARD methods in accuracy and earliness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18292v1">Measuring Fine-Grained Negotiation Tactics of Humans and LLMs in Diplomacy</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      The study of negotiation styles dates back to Aristotle's ethos-pathos-logos rhetoric. Prior efforts primarily studied the success of negotiation agents. Here, we shift the focus towards the styles of negotiation strategies. Our focus is the strategic dialogue board game Diplomacy, which affords rich natural language negotiation and measures of game success. We used LLM-as-a-judge to annotate a large human-human set of Diplomacy games for fine-grained negotiation tactics from a sociologically-grounded taxonomy. Using a combination of the It Takes Two and WebDiplomacy datasets, we demonstrate the reliability of our LLM-as-a-Judge framework and show strong correlations between negotiation features and success in the Diplomacy setting. Lastly, we investigate the differences between LLM and human negotiation strategies and show that fine-tuning can steer LLM agents toward more human-like negotiation behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18265v1">Intelligent Human-Machine Partnership for Manufacturing: Enhancing Warehouse Planning through Simulation-Driven Knowledge Graphs and LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 13 pages, 2 figures, accepted for oral presentation at AAAI Human Machine Collaboration Workshop 2026
    </div>
    <details class="paper-abstract">
      Manufacturing planners face complex operational challenges that require seamless collaboration between human expertise and intelligent systems to achieve optimal performance in modern production environments. Traditional approaches to analyzing simulation-based manufacturing data often create barriers between human decision-makers and critical operational insights, limiting effective partnership in manufacturing planning. Our framework establishes a collaborative intelligence system integrating Knowledge Graphs and Large Language Model-based agents to bridge this gap, empowering manufacturing professionals through natural language interfaces for complex operational analysis. The system transforms simulation data into semantically rich representations, enabling planners to interact naturally with operational insights without specialized expertise. A collaborative LLM agent works alongside human decision-makers, employing iterative reasoning that mirrors human analytical thinking while generating precise queries for knowledge extraction and providing transparent validation. This partnership approach to manufacturing bottleneck identification, validated through operational scenarios, demonstrates enhanced performance while maintaining human oversight and decision authority. For operational inquiries, the system achieves near-perfect accuracy through natural language interaction. For investigative scenarios requiring collaborative analysis, we demonstrate the framework's effectiveness in supporting human experts to uncover interconnected operational issues that enhance understanding and decision-making. This work advances collaborative manufacturing by creating intuitive methods for actionable insights, reducing cognitive load while amplifying human analytical capabilities in evolving manufacturing ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11733v2">SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 AAAI-2026 poster; 7 pages for main content, 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      LLM-based multi-agent systems exhibit strong collaborative capabilities but often suffer from redundant communication and excessive token overhead. Existing methods typically enhance efficiency through pretrained GNNs or greedy algorithms, but often isolate pre- and post-task optimization, lacking a unified strategy. To this end, we present SafeSieve, a progressive and adaptive multi-agent pruning algorithm that dynamically refines the inter-agent communication through a novel dual-mechanism. SafeSieve integrates initial LLM-based semantic evaluation with accumulated performance feedback, enabling a smooth transition from heuristic initialization to experience-driven refinement. Unlike existing greedy Top-k pruning methods, SafeSieve employs 0-extension clustering to preserve structurally coherent agent groups while eliminating ineffective links. Experiments across benchmarks (SVAMP, HumanEval, etc.) showcase that SafeSieve achieves 94.01% average accuracy while reducing token usage by 12.4%-27.8%. Results further demonstrate robustness under prompt injection attacks (1.23% average accuracy drop). In heterogeneous settings, SafeSieve reduces deployment costs by 13.3% while maintaining performance. These results establish SafeSieve as an efficient, GPU-free, and scalable framework for practical multi-agent systems. Our code can be found here: https://github.com/csgen/SafeSieve
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18196v1">Training LLMs with LogicReward for Faithful and Rigorous Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Although LLMs exhibit strong reasoning capabilities, existing training methods largely depend on outcome-based feedback, which can produce correct answers with flawed reasoning. Prior work introduces supervision on intermediate steps but still lacks guarantees of logical soundness, which is crucial in high-stakes scenarios where logical consistency is paramount. To address this, we propose LogicReward, a novel reward system that guides model training by enforcing step-level logical correctness with a theorem prover. We further introduce Autoformalization with Soft Unification, which reduces natural language ambiguity and improves formalization quality, enabling more effective use of the theorem prover. An 8B model trained on data constructed with LogicReward surpasses GPT-4o and o4-mini by 11.6\% and 2\% on natural language inference and logical reasoning tasks with simple training procedures. Further analysis shows that LogicReward enhances reasoning faithfulness, improves generalizability to unseen tasks such as math and commonsense reasoning, and provides a reliable reward signal even without ground-truth labels. We will release all data and code at https://llm-symbol.github.io/LogicReward.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18194v1">TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Disaggregated LLM serving improves resource efficiency by separating the compute-intensive prefill phase from the latency-critical decode phase. However, this architecture introduces a fundamental bottleneck: key/value (KV) tensors generated during prefill must be transferred to decode workers, and existing systems rely on RDMA-based network paths for this exchange. As model sizes and context lengths increase, KV transfer dominates both time-to-first-token (TTFT) and peak throughput, and remains highly sensitive to network contention even when prefix reuse is high. This paper presents TraCT, a rack-scale LLM serving system that uses CXL shared memory as both a KV-transfer substrate and a rack-wide prefix-aware KV cache. TraCT enables GPUs to write and read KV blocks directly through CXL load/store and DMA operations, eliminating the NIC hop that constrains existing disaggregated pipelines. However, to realize this design, multiple new challenges such as synchronization, consistency, and data management on non-coherent CXL memory need to be addressed. TraCT proposes various software solutions such as the two-tier inter-node synchronization mechanism to address these challenges. We implement TraCT on the Dynamo LLM inference framework and show that, across static and synthetic workloads, TraCT reduces average TTFT by up to 9.8x, lowers P99 latency by up to 6.2x, and improves peak throughput by up to 1.6x compared to RDMA and DRAM-based caching baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02230v2">Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      KV cache management is essential for efficient LLM inference. To maximize utilization, existing inference engines evict finished requests' KV cache if new requests are waiting. This policy breaks for agentic workloads, which interleave LLM calls with tools, introducing pauses that prevent effective KV reuse across turns. Since some tool calls have much shorter durations than human response multi-turn chatbot, it would be promising to retain the KV cache in during these tools. However, there are many challenges. First, we need to consider both the potential cost of recomputation or reloading (if CPU offloading enabled) and the increasing queueing delays after eviction from GPU. Second, due to the internal variance of tool call durations, we need the method to remain robust under limited predictability of tool call durations. We present Continuum, a serving system to optimize job completion time for multi-turn agent workloads by introducing time-to-live mechanism for KV cache retaining. For LLM request that generates a tool call, Continuum selectively pins the KV cache in GPU memory with a time-to-live value determined by considering both the reload cost and ordering preserve benefit of retaining KV cache. Moreover, when the TTL expires, the KV cache can be automatically evicted to free up GPU memory, providing robust performance under edge cases. When combined with program-level first-come-first-serve, Continuum preserves multi-turn continuity, and reduces delay for complex agentic workflows. Our evaluation on real-world agentic workloads (SWE-Bench and BFCL) with Llama-3.1 8B/70B shows that Continuum significantly improves the average job completion times and its improvement scales with turn number increase. We release a preview version at: https://github.com/Hanchenli/vllm-continuum
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.10321v3">Towards Human-Guided, Data-Centric LLM Co-Pilots</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Saveliev, Liu & Seedat contributed equally
    </div>
    <details class="paper-abstract">
      Machine learning (ML) has the potential to revolutionize various domains, but its adoption is often hindered by the disconnect between the needs of domain experts and translating these needs into robust and valid ML tools. Despite recent advances in LLM-based co-pilots to democratize ML for non-technical domain experts, these systems remain predominantly focused on model-centric aspects while overlooking critical data-centric challenges. This limitation is problematic in complex real-world settings where raw data often contains complex issues, such as missing values, label noise, and domain-specific nuances requiring tailored handling. To address this we introduce CliMB-DC, a human-guided, data-centric framework for LLM co-pilots that combines advanced data-centric tools with LLM-driven reasoning to enable robust, context-aware data processing. At its core, CliMB-DC introduces a novel, multi-agent reasoning system that combines a strategic coordinator for dynamic planning and adaptation with a specialized worker agent for precise execution. Domain expertise is then systematically incorporated to guide the reasoning process using a human-in-the-loop approach. To guide development, we formalize a taxonomy of key data-centric challenges that co-pilots must address. Thereafter, to address the dimensions of the taxonomy, we integrate state-of-the-art data-centric tools into an extensible, open-source architecture, facilitating the addition of new tools from the research community. Empirically, using real-world healthcare datasets we demonstrate CliMB-DC's ability to transform uncurated datasets into ML-ready formats, significantly outperforming existing co-pilot baselines for handling data-centric challenges. CliMB-DC promises to empower domain experts from diverse domains -- healthcare, finance, social sciences and more -- to actively participate in driving real-world impact using ML.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v4">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17814v1">LLM-based Behaviour Driven Development for Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 7 pages, keynote given at 2nd International Symposium on Artificial Intelligence and Internet of Things (AIIoT-25), December 22-24th, 2025
    </div>
    <details class="paper-abstract">
      Test and verification are essential activities in hardware and system design, but their complexity grows significantly with increasing system sizes. While Behavior Driven Development (BDD) has proven effective in software engineering, it is not yet well established in hardware design, and its practical use remains limited. One contributing factor is the manual effort required to derive precise behavioral scenarios from textual specifications. Recent advances in Large Language Models (LLMs) offer new opportunities to automate this step. In this paper, we investigate the use of LLM-based techniques to support BDD in the context of hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20150v4">Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at https://github.com/yaochenzhu/Rank-GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09147v4">LLM-as-a-qualitative-judge: automating error analysis in natural language generation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that instance-specific issues output by LLM-as-a-qualitative-judge match those annotated by humans in 2/3 cases, and that LLM-as-a-qualitative-judge is capable of producing error type reports resembling the reports composed by human annotators. We also demonstrate in a case study how the use of LLM-as-a-qualitative-judge can substantially improve NLG systems performance. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12817v2">Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted as a conference paper at AAAI'26
    </div>
    <details class="paper-abstract">
      The recent proliferation of large language models (LLMs) holds the potential to revolutionize healthcare, with strong capabilities in diverse medical tasks. Yet, deploying LLMs in high-stakes healthcare settings requires rigorous verification and validation to understand any potential harm. This paper investigates the reliability and viability of using medical knowledge graphs (KGs) for the automated factuality evaluation of LLM-generated responses. To ground this investigation, we introduce FAITH, a framework designed to systematically probe the strengths and limitations of this KG-based approach. FAITH operates without reference answers by decomposing responses into atomic claims, linking them to a medical KG, and scoring them based on evidence paths. Experiments on diverse medical tasks with human subjective evaluations demonstrate that KG-grounded evaluation achieves considerably higher correlations with clinician judgments and can effectively distinguish LLMs with varying capabilities. It is also robust to textual variances. The inherent explainability of its scoring can further help users understand and mitigate the limitations of current LLMs. We conclude that while limitations exist, leveraging KGs is a prominent direction for automated factuality assessment in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.21540v3">Designing an LLM-Based Behavioral Activation Chatbot for Young People with Depression: Insights from an Evaluation with Artificial Users and Clinical Experts</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      LLMs promise to overcome limitations of rule-based mental health chatbots through improved natural language capabilities, yet their ability to deliver evidence-based psychological interventions remains largely unverified because evaluations rarely apply the validated fidelity measures used to assess psychotherapists. We developed an LLM-based chatbot that delivers behavioral activation for depression and generated 48 complete chat sessions with diverse artificial users. Ten psychotherapists assessed these sessions using the Quality of Behavioral Activation Scale (Q-BAS), a validated fidelity instrument. Results show that the chatbot reliably executed the intervention across all phases and maintained safety protocols, but it struggled with clinical judgment, particularly when verifying the feasibility of proposed activities. Overall, our findings suggest that LLM-based chatbots can execute therapeutic protocols with high fidelity, while robust clinical reasoning remains an open challenge. We outline design implications to address this gap and provide the chatbot and artificial user prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17639v1">Linear Personality Probing and Steering in LLMs: A Big Five Study</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 29 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit distinct and consistent personalities that greatly impact trust and engagement. While this means that personality frameworks would be highly valuable tools to characterize and control LLMs' behavior, current approaches remain either costly (post-training) or brittle (prompt engineering). Probing and steering via linear directions has recently emerged as a cheap and efficient alternative. In this paper, we investigate whether linear directions aligned with the Big Five personality traits can be used for probing and steering model behavior. Using Llama 3.3 70B, we generate descriptions of 406 fictional characters and their Big Five trait scores. We then prompt the model with these descriptions and questions from the Alpaca questionnaire, allowing us to sample hidden activations that vary along personality traits in known, quantifiable ways. Using linear regression, we learn a set of per-layer directions in activation space, and test their effectiveness for probing and steering model behavior. Our results suggest that linear directions aligned with trait-scores are effective probes for personality detection, while their steering capabilities strongly depend on context, producing reliable effects in forced-choice tasks but limited influence in open-ended generation or when additional context is present in the prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17630v1">Confidence-Credibility Aware Weighted Ensembles of Small LLMs Outperform Large LLMs in Emotion Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at IRICT 2025
    </div>
    <details class="paper-abstract">
      This paper introduces a confidence-weighted, credibility-aware ensemble framework for text-based emotion detection, inspired by Condorcet's Jury Theorem (CJT). Unlike conventional ensembles that often rely on homogeneous architectures, our approach combines architecturally diverse small transformer-based large language models (sLLMs) - BERT, RoBERTa, DistilBERT, DeBERTa, and ELECTRA, each fully fine-tuned for emotion classification. To preserve error diversity, we minimize parameter convergence while taking advantage of the unique biases of each model. A dual-weighted voting mechanism integrates both global credibility (validation F1 score) and local confidence (instance-level probability) to dynamically weight model contributions. Experiments on the DAIR-AI dataset demonstrate that our credibility-confidence ensemble achieves a macro F1 score of 93.5 percent, surpassing state-of-the-art benchmarks and significantly outperforming large-scale LLMs, including Falcon, Mistral, Qwen, and Phi, even after task-specific Low-Rank Adaptation (LoRA). With only 595M parameters in total, our small LLMs ensemble proves more parameter-efficient and robust than models up to 7B parameters, establishing that carefully designed ensembles of small, fine-tuned models can outperform much larger LLMs in specialized natural language processing (NLP) tasks such as emotion detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17570v1">GreedySnake: Accelerating SSD-Offloaded LLM Training with Efficient Scheduling and Optimizer Step Overlapping</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      SSD-offloaded training offers a practical and promising approach to making LLM training cost-effective. Building on gradient accumulation with micro-batches, this paper introduces GreedySnake, a new SSD-offloaded training system that employs vertical scheduling, which executes all microbatches of a layer before proceeding to the next. Compared to existing systems that use horizontal scheduling (i.e., executing micro-batches sequentially), GreedySnake achieves higher training throughput with smaller batch sizes, bringing the system much closer to the ideal scenario predicted by the roofline model. To further mitigate the I/O bottleneck, GreedySnake overlaps part of the optimization step with the forward pass of the next iteration. Experimental results on A100 GPUs show that GreedySnake achieves saturated training throughput improvements over ZeRO-Infinity: 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B. The code is open-sourced at https://github.com/npz7yyk/GreedySnake
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17540v1">SGCR: A Specification-Grounded Framework for Trustworthy LLM Code Review</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automating code review with Large Language Models (LLMs) shows immense promise, yet practical adoption is hampered by their lack of reliability, context-awareness, and control. To address this, we propose Specification-Grounded Code Review (SGCR), a framework that grounds LLMs in human-authored specifications to produce trustworthy and relevant feedback. SGCR features a novel dual-pathway architecture: an explicit path ensures deterministic compliance with predefined rules derived from these specifications, while an implicit path heuristically discovers and verifies issues beyond those rules. Deployed in a live industrial environment at HiThink Research, SGCR's suggestions achieved a 42% developer adoption rate-a 90.9% relative improvement over a baseline LLM (22%). Our work demonstrates that specification-grounding is a powerful paradigm for bridging the gap between the generative power of LLMs and the rigorous reliability demands of software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.18686v2">Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Time series classification plays a fundamental role in a wide range of real-world applications. Recently, large language models (LLMs) have demonstrated strong generalization and reasoning capacities, but directly applying them to time series classification remains non-trivial due to the representation gap between numerical sequences and linguistic semantics. In this paper, we propose HiTime, a hierarchical LLM-based framework for multimodal time series classification that bridges structured temporal representations with semantic reasoning in a generative paradigm. Specifically, we design a hierarchical sequence feature encoding module composed of a data-specific encoder and a task-specific encoder to extract complementary temporal features. To mitigate the embedding gap between time series representations and textual semantics, we further introduce a semantic space alignment module that jointly performs coarse-grained global modeling and fine-grained cross-modal correspondence. Building upon the above representations, we employ a parameter-efficient supervised fine-tuning strategy to activate the generative classification capability of the algined LLMs, thereby transforming conventional discriminative time series classification into a generative task. Extensive experiments on multiple benchmarks demonstrate that the proposed framework consistently outperforms state-of-the-art baselines. The code is publicly available at https://github.com/Xiaoyu-Tao/HiTime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17287v3">RBCTest: Leveraging LLMs to Mine and Verify Oracles of API Response Bodies for RESTful API Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 The paper is accepted to ICSE 2026
    </div>
    <details class="paper-abstract">
      In API testing, deriving logical constraints on API response bodies to be used as oracles is crucial for generating test cases and performing automated testing of RESTful APIs. However, existing approaches are restricted to dynamic analysis, in which oracles are extracted via the execution of APIs as part of the system under test. In this paper, we propose a complementary LLM-based static approach in which constraints for API response bodies are mined from API specifications. We leverage large language models (LLMs) to comprehend API specifications, mine constraints for response bodies, and generate test cases. To reduce LLM hallucination, we apply an Observation-Confirmation (OC) scheme that uses initial prompts to contextualize constraints, allowing subsequent prompts to more accurately confirm their presence. Our empirical results show that RBCTest with OC prompting achieves high precision in constraint mining, with averages ranging from 85.1% to 93.6%. It also performs well in generating test cases from mined constraints, with precision ranging from 86.4% to 91.7%. We further use test cases generated by RBCTest to detect 46 mismatches between API specifications and actual response data across 19 real-world APIs. Four of these mismatches were reported in developers' forums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17375v1">AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Reward models and LLM-as-a-Judge systems are central to modern post-training pipelines such as RLHF, DPO, and RLAIF, where they provide scalar feedback and binary decisions that guide model selection and RL-based fine-tuning. We show that these judge systems exhibit a recurring vulnerability: short sequences of low-perplexity control tokens can flip many binary evaluations from correct ``No'' judgments to incorrect ``Yes'' judgments by steering the last-layer logit gap. These control tokens are patterns that a policy model could plausibly generate during post-training, and thus represent realistic reward-hacking risks rather than worst-case adversarial strings. Our method, AdvJudge-Zero, uses the model's next-token distribution and beam-search exploration to discover diverse control-token sequences from scratch, and our analysis shows that the induced hidden-state perturbations concentrate in a low-rank ``soft mode'' that is anti-aligned with the judge's refusal direction. Empirically, these tokens cause very high false positive rates when large open-weight and specialized judge models score incorrect answers on math and reasoning benchmarks. Finally, we show that LoRA-based adversarial training on small sets of control-token-augmented examples can markedly reduce these false positives while preserving evaluation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25123v3">From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17334v1">Bridging Natural Language and Formal Specification--Automated Translation of Software Requirements to LTL via Hierarchical Semantics Decomposition Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automating the translation of natural language (NL) software requirements into formal specifications remains a critical challenge in scaling formal verification practices to industrial settings, particularly in safety-critical domains. Existing approaches, both rule-based and learning-based, face significant limitations. While large language models (LLMs) like GPT-4o demonstrate proficiency in semantic extraction, they still encounter difficulties in addressing the complexity, ambiguity, and logical depth of real-world industrial requirements. In this paper, we propose Req2LTL, a modular framework that bridges NL and Linear Temporal Logic (LTL) through a hierarchical intermediate representation called OnionL. Req2LTL leverages LLMs for semantic decomposition and combines them with deterministic rule-based synthesis to ensure both syntactic validity and semantic fidelity. Our comprehensive evaluation demonstrates that Req2LTL achieves 88.4% semantic accuracy and 100% syntactic correctness on real-world aerospace requirements, significantly outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v2">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 14 pages, 20 figures, conference
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16882v2">Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14121v2">SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances in Large Language Models (LLMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by contrasting the keyframes with the target models. Finally, we propose SportsRAG, a RAG-based training guidance model built upon Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v2">Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17259v1">Verifiability-First Agents: Provable Observability and Lightweight Audit Agents for Controlling Autonomous LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      As LLM-based agents grow more autonomous and multi-modal, ensuring they remain controllable, auditable, and faithful to deployer intent becomes critical. Prior benchmarks measured the propensity for misaligned behavior and showed that agent personalities and tool access significantly influence misalignment. Building on these insights, we propose a Verifiability-First architecture that (1) integrates run-time attestations of agent actions using cryptographic and symbolic methods, (2) embeds lightweight Audit Agents that continuously verify intent versus behavior using constrained reasoning, and (3) enforces challenge-response attestation protocols for high-risk operations. We introduce OPERA (Observability, Provable Execution, Red-team, Attestation), a benchmark suite and evaluation protocol designed to measure (i) detectability of misalignment, (ii) time to detection under stealthy strategies, and (iii) resilience of verifiability mechanisms to adversarial prompt and persona injection. Our approach shifts the evaluation focus from how likely misalignment is to how quickly and reliably misalignment can be detected and remediated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17247v1">Incorporating Error Level Noise Embedding for Improving LLM-Assisted Robustness in Persian Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) systems suffer significant performance degradation in noisy environments, a challenge that is especially severe for low-resource languages such as Persian. Even state-of-the-art models such as Whisper struggle to maintain accuracy under varying signal-to-noise ratios (SNRs). This study presents a robust noise-sensitive ASR error correction framework that combines multiple hypotheses and noise-aware modeling. Using noisy Persian speech, we generate 5-best hypotheses from a modified Whisper-large decoder. Error Level Noise (ELN) is introduced as a representation that captures semantic- and token-level disagreement across hypotheses, quantifying the linguistic distortions caused by noise. ELN thus provides a direct measure of noise-induced uncertainty, enabling the LLM to reason about the reliability of each hypothesis during correction. Three models are evaluated: (1) a base LLaMA-2-7B model without fine-tuning, (2) a fine-tuned variant trained on text-only hypotheses, and (3) a noise-conditioned model integrating ELN embeddings at both sentence and word levels. Experimental results demonstrate that the ELN-conditioned model achieves substantial reductions in Word Error Rate (WER). Specifically, on the challenging Mixed Noise test set, the proposed Fine-tuned + ELN (Ours) model reduces the WER from a baseline of 31.10\% (Raw Whisper) to 24.84\%, significantly surpassing the Fine-tuned (No ELN) text-only baseline of 30.79\%, whereas the original LLaMA-2-7B model increased the WER to 64.58\%, demonstrating that it is unable to correct Persian errors on its own. This confirms the effectiveness of combining multiple hypotheses with noise-aware embeddings for robust Persian ASR in noisy real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10689v2">Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: https://lcowiclr2025.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v4">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF, Camera ready version
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17172v1">PILAR: Personalizing Augmented Reality Interactions with LLM-based Human-Centric and Trustworthy Explanations for Daily Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Published in the 2025 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI)-driven augmented reality (AR) systems are becoming increasingly integrated into daily life, and with this growth comes a greater need for explainability in real-time user interactions. Traditional explainable AI (XAI) methods, which often rely on feature-based or example-based explanations, struggle to deliver dynamic, context-specific, personalized, and human-centric insights for everyday AR users. These methods typically address separate explainability dimensions (e.g., when, what, how) with different explanation techniques, resulting in unrealistic and fragmented experiences for seamless AR interactions. To address this challenge, we propose PILAR, a novel framework that leverages a pre-trained large language model (LLM) to generate context-aware, personalized explanations, offering a more intuitive and trustworthy experience in real-time AI-powered AR systems. Unlike traditional methods, which rely on multiple techniques for different aspects of explanation, PILAR employs a unified LLM-based approach that dynamically adapts explanations to the user's needs, fostering greater trust and engagement. We implement the PILAR concept in a real-world AR application (e.g., personalized recipe recommendations), an open-source prototype that integrates real-time object detection, recipe recommendation, and LLM-based personalized explanations of the recommended recipes based on users' dietary preferences. We evaluate the effectiveness of PILAR through a user study with 16 participants performing AR-based recipe recommendation tasks, comparing an LLM-based explanation interface to a traditional template-based one. Results show that the LLM-based interface significantly enhances user performance and experience, with participants completing tasks 40% faster and reporting greater satisfaction, ease of use, and perceived transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06000v2">LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Review Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at AACL 2025 Version 2 Updated with Final version
    </div>
    <details class="paper-abstract">
      Clinical interventions often hinge on age: medications and procedures safe for adults may be harmful to children or ineffective for older adults. However, as language models are increasingly integrated into biomedical evidence synthesis workflows, it remains uncertain whether these systems preserve such crucial demographic distinctions. To address this gap, we evaluate how well state-of-the-art language models retain age-related information when generating abstractive summaries of biomedical studies. We construct DemogSummary, a novel age-stratified dataset of systematic review primary studies, covering child, adult, and older adult populations. We evaluate three prominent summarisation-capable LLMs, Qwen (open-source), Longformer (open-source) and GPT-4.1 Nano (proprietary), using both standard metrics and a newly proposed Demographic Salience Score (DSS), which quantifies age-related entity retention and hallucination. Our results reveal systematic disparities across models and age groups: demographic fidelity is lowest for adult-focused summaries, and under-represented populations are more prone to hallucinations. These findings highlight the limitations of current LLMs in faithful and bias-free summarisation and point to the need for fairness-aware evaluation frameworks and summarisation pipelines in biomedical NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11027v4">On the Effect of Sampling Diversity in Scaling LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) scaling inference is key to unlocking greater performance, and leveraging diversity has proven an effective way to enhance it. Motivated by the observed relationship between solution accuracy and meaningful response diversity, we systematically study the effect of prompt diversity in scaling inference. We theoretically explain why diversified sampling improves Best-of-N scaling, showing that responses generated from diverse prompts after Best-of-N selection exhibit significantly lower error rates than those produced from stationary prompts. Building on this analysis, we derive a diversity-fidelity trade-off principle, that guides the design of sampling strategies introducing diversity. From this guidance, we instantiate a family of effective perturbation styles. We theoretically and empirically characterize when diversified exploration remains effective, demonstrating that it works under a variety of conditions, and we further show that under majority voting, diversity may vanish. Finally, we systematically evaluate how effective sampling diversity is and show that, when applied appropriately in different contexts, it yields relative gains of 10.8% in EM@100 for reasoning, 9.6% for mathematics, and 9.5% in Pass@100 for code generation. Overall, this work provides a systematic analysis that offers a theoretical and empirical foundation for understanding how sampling diversity affects LLM inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17145v1">Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 10 pages, ACRA 2025, Submitted, Accepted and Presented
    </div>
    <details class="paper-abstract">
      Reasoning under uncertainty is a key challenge in AI, especially for real-world tasks, where problems with sparse data demands systematic generalisation. Existing approaches struggle to balance accuracy and simplicity when evaluating multiple candidate solutions. We propose a Solomonoff-inspired method that weights LLM-generated hypotheses by simplicity and predictive fit. Applied to benchmark (Mini-ARC) tasks, our method produces Solomonoff-weighted mixtures for per-cell predictions, yielding conservative, uncertainty-aware outputs even when hypotheses are noisy or partially incorrect. Compared to Bayesian Model Averaging (BMA), Solomonoff scoring spreads probability more evenly across competing hypotheses, while BMA concentrates weight on the most likely but potentially flawed candidates. Across tasks, this highlights the value of algorithmic information-theoretic priors for interpretable, reliable multi-hypothesis reasoning under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20196v2">Understanding and supporting how developers prompt for LLM-powered code editing in practice</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming software engineering, with coding assistants embedded in an IDE becoming increasingly prevalent. While research has focused on improving the tools and understanding developer perceptions, a critical gap exists in understanding how developers actually use these tools in their daily workflows, and, crucially, where they struggle. This paper addresses part of this gap through a multi-phased investigation of developer interactions with an LLM-powered code editing feature, Transform Code, in an IDE widely used at Google. First, we analyze telemetry logs of the feature usage, revealing that frequent re-prompting can be an indicator of developer struggles with using Transform Code. Second, we conduct a qualitative analysis of unsatisfactory requests, identifying five key categories of information often missing from developer prompts. Finally, based on these findings, we propose and evaluate a tool, AutoPrompter, for automatically improving prompts by inferring missing information from the surrounding code context, leading to a 27% improvement in edit correctness on our test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10729v3">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 NeurIPS Workshop on Learning from Time Series for Health
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18131v1">Holistic Evaluation of State-of-the-Art LLMs for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 13 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This study presents a comprehensive empirical evaluation of six state-of-the-art large language models (LLMs) for code generation, including both general-purpose and code-specialized models. Using a dataset of 944 real-world LeetCode problems across five programming languages, we assess model performance using rigorous metrics: compile-time errors, runtime errors, functional failures, and algorithmic suboptimalities. The results reveal significant performance variations, with DeepSeek-R1 and GPT-4.1 consistently outperform others in terms of correctness, efficiency, and robustness. Through detailed case studies, we identify common failure scenarios such as syntax errors, logical flaws, and suboptimal algorithms, highlighting the critical role of prompt engineering and human oversight in improving results. Based on these findings, we provide actionable recommendations for developers and practitioners, emphasizing that successful LLM deployment depends on careful model selection, effective prompt design, and context-aware usage to ensure reliable code generation in real-world software development tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22732v2">WebATLAS: An LLM Agent with Experience-Driven Memory and Action Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 9 pages, NeurIPS 2025 Workshop on Language Agents and World Models
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) web agents often struggle with long-horizon web navigation and web task completion in new websites, producing inefficient action sequences unless fine-tuned on environment-specific data. We show that experience-driven memory, combined with look-ahead action simulation, is sufficient for LLM agents to adapt to unseen web environments by remembering past failures and predicting the consequences of future actions. We introduce WebATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented LLM web agent that learns a lightweight internal model of the environment from interaction experience and performs hypothetical action rollouts before acting in the real world. WebATLAS builds a persistent cognitive map via curiosity-driven exploration, stores interaction outcomes as experience-based memory, and evaluates candidate actions in cognitive space using a planner--simulator--critic loop. This enables the agent to reuse past experience, avoid previously unsuccessful behaviors, and generate more efficient plans. We evaluate WebATLAS on the WebArena-Lite benchmark for autonomous web navigation and demonstrate a success rate of 63%, outperforming the previous state-of-the-art at 53.9%. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablation studies confirm that experience-driven memory, look-ahead action simulation, and hierarchical replanning play complementary roles in enabling robust, training-free web agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13901v2">RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14918v2">Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14625v2">Accelerated Preference Elicitation with LLM-Based Proxies</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 In Proceedings of The 21st Conference on Web and Internet Economics (WINE '25)
    </div>
    <details class="paper-abstract">
      Bidders in combinatorial auctions face significant challenges when describing their preferences to an auctioneer. Classical work on preference elicitation focuses on query-based techniques inspired from proper learning--often via proxies that interface between bidders and an auction mechanism--to incrementally learn bidder preferences as needed to compute efficient allocations. Although such elicitation mechanisms enjoy theoretical query efficiency, the amount of communication required may still be too cognitively taxing in practice. We propose a family of efficient LLM-based proxy designs for eliciting preferences from bidders using natural language. Our proposed mechanism combines LLM pipelines and DNF-proper-learning techniques to quickly approximate preferences when communication is limited. To validate our approach, we create a testing sandbox for elicitation mechanisms that communicate in natural language. In our experiments, our most promising LLM proxy design reaches approximately efficient outcomes with five times fewer queries than classical proper learning based elicitation mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18020v1">Specification and Detection of LLM Code Smells</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted paper at ICSE NIER 2026 : https://conf.researchr.org/track/icse-2026/icse-2026-nier
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained massive popularity in recent years and are increasingly integrated into software systems for diverse purposes. However, poorly integrating them in source code may undermine software system quality. Yet, to our knowledge, there is no formal catalog of code smells specific to coding practices for LLM inference. In this paper, we introduce the concept of LLM code smells and formalize five recurrent problematic coding practices related to LLM inference in software systems, based on relevant literature. We extend the detection tool SpecDetect4AI to cover the newly defined LLM code smells and use it to validate their prevalence in a dataset of 200 open-source LLM systems. Our results show that LLM code smells affect 60.50% of the analyzed systems, with a detection precision of 86.06%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17970v1">CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Weight-only quantization is widely used to mitigate the memory-bound nature of LLM inference. Codebook-based methods extend this trend by achieving strong accuracy in the extremely low-bit regime (e.g., 2-bit). However, current kernels rely on dequantization, which repeatedly fetches centroids and reconstructs weights, incurring substantial latency and cache pressure. We present CodeGEMM, a codebook-centric GEMM kernel that replaces dequantization with precomputed inner products between centroids and activations stored in a lightweight Psumbook. At inference, code indices directly gather these partial sums, eliminating per-element lookups and reducing the on-chip footprint. The kernel supports the systematic exploration of latency-memory-accuracy trade-offs under a unified implementation. On Llama-3 models, CodeGEMM delivers 1.83x (8B) and 8.93x (70B) speedups in the 2-bit configuration compared to state-of-the-art codebook-based quantization at comparable accuracy and further improves computing efficiency and memory subsystem utilization.
    </details>
</div>
