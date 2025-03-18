# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06145v3">Escalating LLM-based Code Translation Benchmarking into the Class-level Era</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have dramatically advanced the performance of automated code translation, making their computational accuracy score reach up to over 80% on many previous benchmarks. However, most code samples in these benchmarks are short, standalone, statement/method-level, and algorithmic, which is not aligned with practical coding tasks. Therefore, it is still unknown the actual capability of LLMs in translating code samples written for daily development. To achieve this, we construct a class-level code translation benchmark, ClassEval-T, and make the first attempt to extensively assess recent LLMs' performance on class-level code translation. ClassEval-T is extended from ClassEval, a well-known class-level Python code generation benchmark consisting of multiple practical coding topics, such as database operation and game design, and diverse contextual dependencies (e.g., fields, methods, and libraries). It cost us 360 person-hours to accomplish the manual migration to Java and C++ with complete code samples and associated test suites. Subsequently, we design three translation strategies (i.e., holistic, min-dependency, and standalone) for class-level code translations and evaluate eight recent LLMs of commercial, general, and code kinds in diverse families and sizes on ClassEval-T. Experimental results demonstrate a remarkable performance drop compared with the most widely studied method-level code translation benchmark, and obvious discrepancies among LLMs appear, showing the effectiveness of ClassEval-T in measuring recent LLMs. Afterwards, we further discuss the usage scenarios for diverse translation strategies and LLMs' ability to dependency awareness when translating class samples. Finally, 1,243 failure cases made by the best-performing LLM under test are analyzed and categorized in this paper for practical guidance and future enlightenment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02532v1">Use Me Wisely: AI-Driven Assessment for LLM Prompting Skills Development</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Preprint accepted for Publication in Educational Technology & Society (ET&S)
    </div>
    <details class="paper-abstract">
      The use of large language model (LLM)-powered chatbots, such as ChatGPT, has become popular across various domains, supporting a range of tasks and processes. However, due to the intrinsic complexity of LLMs, effective prompting is more challenging than it may seem. This highlights the need for innovative educational and support strategies that are both widely accessible and seamlessly integrated into task workflows. Yet, LLM prompting is highly task- and domain-dependent, limiting the effectiveness of generic approaches. In this study, we explore whether LLM-based methods can facilitate learning assessments by using ad-hoc guidelines and a minimal number of annotated prompt samples. Our framework transforms these guidelines into features that can be identified within learners' prompts. Using these feature descriptions and annotated examples, we create few-shot learning detectors. We then evaluate different configurations of these detectors, testing three state-of-the-art LLMs and ensembles. We run experiments with cross-validation on a sample of original prompts, as well as tests on prompts collected from task-naive learners. Our results show how LLMs perform on feature detection. Notably, GPT- 4 demonstrates strong performance on most features, while closely related models, such as GPT-3 and GPT-3.5 Turbo (Instruct), show inconsistent behaviors in feature classification. These differences highlight the need for further research into how design choices impact feature selection and prompt detection. Our findings contribute to the fields of generative AI literacy and computer-supported learning assessment, offering valuable insights for both researchers and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06860v2">Balancing Efficiency and Effectiveness: An LLM-Infused Approach for Optimized CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 5 pages, 4 figures,4 tables
    </div>
    <details class="paper-abstract">
      Click-Through Rate (CTR) prediction is essential in online advertising, where semantic information plays a pivotal role in shaping user decisions and enhancing CTR effectiveness. Capturing and modeling deep semantic information, such as a user's preference for "H\"aagen-Dazs' HEAVEN strawberry light ice cream" due to its health-conscious and premium attributes, is challenging. Traditional semantic modeling often overlooks these intricate details at the user and item levels. To bridge this gap, we introduce a novel approach that models deep semantic information end-to-end, leveraging the comprehensive world knowledge capabilities of Large Language Models (LLMs). Our proposed LLM-infused CTR prediction framework(Multi-level Deep Semantic Information Infused CTR model via Distillation, MSD) is designed to uncover deep semantic insights by utilizing LLMs to extract and distill critical information into a smaller, more efficient model, enabling seamless end-to-end training and inference. Importantly, our framework is carefully designed to balance efficiency and effectiveness, ensuring that the model not only achieves high performance but also operates with optimal resource utilization. Online A/B tests conducted on the Meituan sponsored-search system demonstrate that our method significantly outperforms baseline models in terms of Cost Per Mile (CPM) and CTR, validating its effectiveness, scalability, and balanced approach in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01747v2">Position: Don't use the CLT in LLM evals with fewer than a few hundred datapoints</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 36 pages, 37 figures
    </div>
    <details class="paper-abstract">
      Rigorous statistical evaluations of large language models (LLMs), including valid error bars and significance testing, are essential for meaningful and reliable performance assessment. Currently, when such statistical measures are reported, they typically rely on the Central Limit Theorem (CLT). In this position paper, we argue that while CLT-based methods for uncertainty quantification are appropriate when benchmarks consist of thousands of examples, they fail to provide adequate uncertainty estimates for LLM evaluations that rely on smaller, highly specialized benchmarks. In these small-data settings, we demonstrate that CLT-based methods perform very poorly, usually dramatically underestimating uncertainty (i.e. producing error bars that are too small). We give recommendations for alternative frequentist and Bayesian methods that are both easy to implement and more appropriate in these increasingly common scenarios. We provide a simple Python library for these Bayesian methods at https://github.com/sambowyer/bayes_evals .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02502v1">LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Submitted to ACL ARR 2024 December
    </div>
    <details class="paper-abstract">
      Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02497v1">PennyLang: Pioneering LLM-Based Quantum Code Generation with a Novel PennyLane-Centric Dataset</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 10 pages, 8 figures, 6 tables, submitted for review under IJCNN 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer remarkable capabilities in code generation, natural language processing, and domain-specific reasoning. Their potential in aiding quantum software development remains underexplored, particularly for the PennyLane framework-a leading platform for hybrid quantum-classical computing. To address this gap, we introduce a novel, high-quality dataset comprising 3,347 PennyLane-specific code samples of quantum circuits and their contextual descriptions, specifically curated to train/fine-tune LLM-based quantum code assistance. Our key contributions are threefold: (1) the automatic creation and open-source release of a comprehensive PennyLane dataset leveraging quantum computing textbooks, official documentation, and open-source repositories; (2) the development of a systematic methodology for data refinement, annotation, and formatting to optimize LLM training efficiency; and (3) a thorough evaluation, based on a Retrieval-Augmented Generation (RAG) framework, demonstrating the effectiveness of our dataset in streamlining PennyLane code generation and improving quantum development workflows. Compared to existing efforts that predominantly focus on Qiskit, our dataset significantly broadens the spectrum of quantum frameworks covered in AI-driven code assistance. By bridging this gap and providing reproducible dataset-creation methodologies, we aim to advance the field of AI-assisted quantum programming, making quantum computing more accessible to both newcomers and experienced developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02463v1">It Helps to Take a Second Opinion: Teaching Smaller LLMs to Deliberate Mutually via Selective Rationale Optimisation</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted at ICLR 2025
    </div>
    <details class="paper-abstract">
      Very large language models (LLMs) such as GPT-4 have shown the ability to handle complex tasks by generating and self-refining step-by-step rationales. Smaller language models (SLMs), typically with < 13B parameters, have been improved by using the data generated from very-large LMs through knowledge distillation. However, various practical constraints such as API costs, copyright, legal and ethical policies restrict using large (often opaque) models to train smaller models for commercial use. Limited success has been achieved at improving the ability of an SLM to explore the space of possible rationales and evaluate them by itself through self-deliberation. To address this, we propose COALITION, a trainable framework that facilitates interaction between two variants of the same SLM and trains them to generate and refine rationales optimized for the end-task. The variants exhibit different behaviors to produce a set of diverse candidate rationales during the generation and refinement steps. The model is then trained via Selective Rationale Optimization (SRO) to prefer generating rationale candidates that maximize the likelihood of producing the ground-truth answer. During inference, COALITION employs a controller to select the suitable variant for generating and refining the rationales. On five different datasets covering mathematical problems, commonsense reasoning, and natural language inference, COALITION outperforms several baselines by up to 5%. Our ablation studies reveal that cross-communication between the two variants performs better than using the single model to self-refine the rationales. We also demonstrate the applicability of COALITION for LMs of varying scales (4B to 14B parameters) and model families (Mistral, Llama, Qwen, Phi). We release the code for this work at https://github.com/Sohanpatnaik106/coalition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02457v1">Don't Get Too Excited -- Eliciting Emotions in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      This paper investigates the challenges of affect control in large language models (LLMs), focusing on their ability to express appropriate emotional states during extended dialogues. We evaluated state-of-the-art open-weight LLMs to assess their affective expressive range in terms of arousal and valence. Our study employs a novel methodology combining LLM-based sentiment analysis with multiturn dialogue simulations between LLMs. We quantify the models' capacity to express a wide spectrum of emotions and how they fluctuate during interactions. Our findings reveal significant variations among LLMs in their ability to maintain consistent affect, with some models demonstrating more stable emotional trajectories than others. Furthermore, we identify key challenges in affect control, including difficulties in producing and maintaining extreme emotional states and limitations in adapting affect to changing conversational contexts. These findings have important implications for the development of more emotionally intelligent AI systems and highlight the need for improved affect modelling in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02450v1">Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at https://github.com/SnowCharmQ/DPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02400v1">Promptware Engineering: Software Engineering for LLM Prompt Development</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software applications, with prompts serving as the primary 'programming' interface to guide their behavior. As a result, a new software paradigm, promptware, has emerged, using natural language prompts to interact with LLMs and enabling complex tasks without traditional coding. Unlike traditional software, which relies on formal programming languages and deterministic runtime environments, promptware is based on ambiguous, unstructured, and context-dependent natural language and operates on LLMs as runtime environments, which are probabilistic and non-deterministic. These fundamental differences introduce unique challenges in prompt development. In practice, prompt development is largely ad hoc and experimental, relying on a time-consuming trial-and-error process - a challenge we term the 'promptware crisis.' To address this, we propose promptware engineering, a new methodology that adapts established software engineering principles to the process of prompt development. Building on decades of success in traditional software engineering, we envision a systematic framework that includes prompt requirements engineering, design, implementation, testing, debugging, and evolution. Unlike traditional software engineering, our framework is specifically tailored to the unique characteristics of prompt development. This paper outlines a comprehensive roadmap for promptware engineering, identifying key research directions and offering actionable insights to advance LLM-based software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03905v2">Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 22 pages, 11 images, 9 tables, Manuscript submitted to a journal (2024)
    </div>
    <details class="paper-abstract">
      LLMs have garnered considerable attention for their potential to streamline Automated Program Repair (APR). LLM-based approaches can either insert the correct code or directly generate patches when provided with buggy methods. However, most of LLM-based APR methods rely on a single type of software information, without fully leveraging different software artifacts. Despite this, many LLM-based approaches do not explore which specific types of information best assist in APR. Addressing this gap is crucial for advancing LLM-based APR techniques. We propose DEVLoRe to use issue content (description and message) and stack error traces to localize buggy methods, then rely on debug information in buggy methods and issue content and stack error to localize buggy lines and generate plausible patches which can pass all unit tests. The results show that while issue content is particularly effective in assisting LLMs with fault localization and program repair, different types of software artifacts complement each other. By incorporating different artifacts, DEVLoRe successfully locates 49.3% and 47.6% of single and non-single buggy methods and generates 56.0% and 14.5% plausible patches for the Defects4J v2.0 dataset, respectively. This outperforms current state-of-the-art APR methods. The source code and experimental results of this work for replication are available at https://github.com/XYZboom/DEVLoRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17102v2">Scholar Name Disambiguation with Search-enhanced LLM Across Language</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      The task of scholar name disambiguation is crucial in various real-world scenarios, including bibliometric-based candidate evaluation for awards, application material anti-fraud measures, and more. Despite significant advancements, current methods face limitations due to the complexity of heterogeneous data, often necessitating extensive human intervention. This paper proposes a novel approach by leveraging search-enhanced language models across multiple languages to improve name disambiguation. By utilizing the powerful query rewriting, intent recognition, and data indexing capabilities of search engines, our method can gather richer information for distinguishing between entities and extracting profiles, resulting in a more comprehensive data dimension. Given the strong cross-language capabilities of large language models(LLMs), optimizing enhanced retrieval methods with this technology offers substantial potential for high-efficiency information retrieval and utilization. Our experiments demonstrate that incorporating local languages significantly enhances disambiguation performance, particularly for scholars from diverse geographic regions. This multi-lingual, search-enhanced methodology offers a promising direction for more efficient and accurate active scholar name disambiguation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02335v1">Unlocking a New Rust Programming Experience: Fast and Slow Thinking with LLMs to Conquer Undefined Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      To provide flexibility and low-level interaction capabilities, the unsafe tag in Rust is essential in many projects, but undermines memory safety and introduces Undefined Behaviors (UBs) that reduce safety. Eliminating these UBs requires a deep understanding of Rust's safety rules and strong typing. Traditional methods require depth analysis of code, which is laborious and depends on knowledge design. The powerful semantic understanding capabilities of LLM offer new opportunities to solve this problem. Although existing large model debugging frameworks excel in semantic tasks, limited by fixed processes and lack adaptive and dynamic adjustment capabilities. Inspired by the dual process theory of decision-making (Fast and Slow Thinking), we present a LLM-based framework called RustBrain that automatically and flexibly minimizes UBs in Rust projects. Fast thinking extracts features to generate solutions, while slow thinking decomposes, verifies, and generalizes them abstractly. To apply verification and generalization results to solution generation, enabling dynamic adjustments and precise outputs, RustBrain integrates two thinking through a feedback mechanism. Experimental results on Miri dataset show a 94.3% pass rate and 80.4% execution rate, improving flexibility and Rust projects safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v2">NitiBench: A Comprehensive Studies of LLM Frameworks Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02328v1">Limited Effectiveness of LLM-based Data Augmentation for COVID-19 Misinformation Stance Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Misinformation surrounding emerging outbreaks poses a serious societal threat, making robust countermeasures essential. One promising approach is stance detection (SD), which identifies whether social media posts support or oppose misleading claims. In this work, we finetune classifiers on COVID-19 misinformation SD datasets consisting of claims and corresponding tweets. Specifically, we test controllable misinformation generation (CMG) using large language models (LLMs) as a method for data augmentation. While CMG demonstrates the potential for expanding training datasets, our experiments reveal that performance gains over traditional augmentation methods are often minimal and inconsistent, primarily due to built-in safeguards within LLMs. We release our code and datasets to facilitate further research on misinformation detection and generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00032v2">Detecting LLM-Generated Korean Text through Linguistic Feature Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02296v1">Memorize or Generalize? Evaluating LLM Code Generation with Evolved Questions</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to exhibit a memorization phenomenon in code generation: instead of truly understanding the underlying principles of a programming problem, they tend to memorize the original prompt and its solution together in the training. Consequently, when facing variants of the original problem, their answers very likely resemble the memorized solutions and fail to generalize. In this paper, we investigate this phenomenon by designing three evolution strategies to create variants: mutation, paraphrasing, and code-rewriting. By comparing the performance and AST similarity of the LLM-generated codes before and after these three evolutions, we develop a memorization score that positively correlates with the level of memorization. As expected, as supervised fine-tuning goes on, the memorization score rises before overfitting, suggesting more severe memorization. We demonstrate that common mitigation approaches, such as prompt translation and using evolved variants as data augmentation in supervised learning and reinforcement learning, either compromise the performance or fail to alleviate the memorization issue. Therefore, memorization remains a significant challenge in LLM code generation, highlighting the need for a more effective solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01048v2">Personalize Your LLM: Fake it then Align it</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is essential for delivering tailored interactions that improve user experience. Many existing personalization methods require fine-tuning LLMs for each user, rendering them prohibitively expensive for widespread adoption. Although retrieval-based approaches offer a more compute-efficient alternative, they still depend on large, high-quality datasets that are not consistently available for all users. To address this challenge, we propose CHAMELEON, a scalable and efficient personalization approach that uses (1) self-generated personal preference data and (2) representation editing to enable quick and cost-effective personalization. Our experiments on various tasks, including those from the LaMP personalization benchmark, show that CHAMELEON efficiently adapts models to personal preferences, improving instruction-tuned models and outperforms two personalization baselines by an average of 40% across two model architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05843v2">Training-free Anomaly Event Detection via LLM-guided Symbolic Pattern Discovery</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Anomaly event detection plays a crucial role in various real-world applications. However, current approaches predominantly rely on supervised learning, which faces significant challenges: the requirement for extensive labeled training data and lack of interpretability in decision-making processes. To address these limitations, we present a training-free framework that integrates open-set object detection with symbolic regression, powered by Large Language Models (LLMs) for efficient symbolic pattern discovery. The LLMs guide the symbolic reasoning process, establishing logical relationships between detected entities. Through extensive experiments across multiple domains, our framework demonstrates several key advantages: (1) achieving superior detection accuracy through direct reasoning without any training process; (2) providing highly interpretable logical expressions that are readily comprehensible to humans; and (3) requiring minimal annotation effort - approximately 1% of the data needed by traditional training-based methods.To facilitate comprehensive evaluation and future research, we introduce two datasets: a large-scale private dataset containing over 110,000 annotated images covering various anomaly scenarios including construction site safety violations, illegal fishing activities, and industrial hazards, along with a public benchmark dataset of 5,000 samples with detailed anomaly event annotations. Code is available at here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02246v1">From Code to Courtroom: LLMs as the New Software Judges</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have been increasingly used to automate SE tasks such as code generation and summarization. However, evaluating the quality of LLM-generated software artifacts remains challenging. Human evaluation, while effective, is very costly and time-consuming. Traditional automated metrics like BLEU rely on high-quality references and struggle to capture nuanced aspects of software quality, such as readability and usefulness. In response, the LLM-as-a-Judge paradigm, which employs LLMs for automated evaluation, has emerged. Given that LLMs are typically trained to align with human judgment and possess strong coding abilities and reasoning skills, they hold promise as cost-effective and scalable surrogates for human evaluators. Nevertheless, LLM-as-a-Judge research in the SE community is still in its early stages, with many breakthroughs needed. This forward-looking SE 2030 paper aims to steer the research community toward advancing LLM-as-a-Judge for evaluating LLMgenerated software artifacts, while also sharing potential research paths to achieve this goal. We provide a literature review of existing SE studies on LLM-as-a-Judge and envision these frameworks as reliable, robust, and scalable human surrogates capable of evaluating software artifacts with consistent, multi-faceted assessments by 2030 and beyond. To validate this vision, we analyze the limitations of current studies, identify key research gaps, and outline a detailed roadmap to guide future developments of LLM-as-a-Judge in software engineering. While not intended to be a definitive guide, our work aims to foster further research and adoption of LLM-as-a-Judge frameworks within the SE community, ultimately improving the effectiveness and scalability of software artifact evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18678v2">Few-shot Personalization of LLMs with Mis-aligned Responses</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 NAACL 25 (main, long), 32 pages
    </div>
    <details class="paper-abstract">
      As the diversity of users increases, the capability of providing personalized responses by large language models (LLMs) has become increasingly important. Existing approaches have only limited successes in LLM personalization, due to the absence of personalized learning or the reliance on shared personal data. This paper proposes a new approach for a few-shot personalization of LLMs with their mis-aligned responses (Fermi). Our key idea is to learn a set of personalized prompts for each user by progressively improving the prompts using LLMs, based on user profile (e.g., demographic information) and a few examples of previous opinions. During an iterative process of prompt improvement, we incorporate the contexts of mis-aligned responses by LLMs, which are especially crucial for the effective personalization of LLMs. In addition, we develop an effective inference method to further leverage the context of the test query and the personalized prompts. Our experimental results demonstrate that Fermi significantly improves performance across various benchmarks, compared to best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13160v2">Understanding Dynamic Diffusion Process of LLM-based Agents under Information Asymmetry</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models have been used to simulate human society using multi-agent systems. Most current social simulation research emphasizes interactive behaviors in fixed environments, ignoring information opacity, relationship variability and diffusion diversity. In this paper, we study the dynamics of information diffusion in 12 asymmetric open environments defined by information content and distribution mechanisms. We first present a general framework to capture the features of information diffusion. Then, we designed a dynamic attention mechanism to help agents allocate attention to different information, addressing the limitations of LLM-based attention. Agents start by responding to external information stimuli within a five-agent group, increasing group size and forming information circles while developing relationships and sharing information. Additionally, we observe the emergence of information cocoons, the evolution of information gaps, and the accumulation of social capital, which are closely linked to psychological, sociological, and communication theories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00061v2">Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 17 pages, 5 figures, 6 tables (NAACL 2025 Findings)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents exhibit remarkable performance across diverse applications by using external tools to interact with environments. However, integrating external tools introduces security risks, such as indirect prompt injection (IPI) attacks. Despite defenses designed for IPI attacks, their robustness remains questionable due to insufficient testing against adaptive attacks. In this paper, we evaluate eight different defenses and bypass all of them using adaptive attacks, consistently achieving an attack success rate of over 50%. This reveals critical vulnerabilities in current defenses. Our research underscores the need for adaptive attack evaluation when designing defenses to ensure robustness and reliability. The code is available at https://github.com/uiuc-kang-lab/AdaptiveAttackAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08600v2">AutoRestTest: A Tool for Automated REST API Testing Using LLMs and MARL</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 To be published in the 47th IEEE/ACM International Conference on Software Engineering - Demonstration Track (ICSE-Demo 2025)
    </div>
    <details class="paper-abstract">
      As REST APIs have become widespread in modern web services, comprehensive testing of these APIs is increasingly crucial. Because of the vast search space of operations, parameters, and parameter values, along with their dependencies and constraints, current testing tools often achieve low code coverage, resulting in suboptimal fault detection. To address this limitation, we present AutoRestTest, a novel tool that integrates the Semantic Property Dependency Graph (SPDG) with Multi-Agent Reinforcement Learning (MARL) and large language models (LLMs) for effective REST API testing. AutoRestTest determines operation-dependent parameters using the SPDG and employs five specialized agents (operation, parameter, value, dependency, and header) to identify dependencies of operations and generate operation sequences, parameter combinations, and values. Through an intuitive command-line interface, users can easily configure and monitor tests with successful operation count, unique server errors detected, and time elapsed. Upon completion, AutoRestTest generates a detailed report highlighting errors detected and operations exercised. In this paper, we introduce our tool and present preliminary findings, with a demonstration video available at https://www.youtube.com/watch?v=VVus2W8rap8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02236v1">VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      In this work, we design and implement VQ-LLM, an efficient fused Vector Quantization (VQ) kernel generation framework. We first introduce a software abstraction called codebook cache to optimize codebook access efficiency and support the integration of VQ with various computations. The codebook cache adaptively stores different entries across the GPU's memory hierarchy, including off-chip global memory, on-chip shared memory, and registers. Centered around the codebook cache, we design an efficient computation engine that optimizes memory traffic during computations involving codebooks. This compute engine adopts the codebook-centric dataflow and fusion optimizations. Additionally, we provide adaptive heuristics to tailor parameter selection in our optimizations to diverse VQ configurations. Our optimizations achieve an average latency reduction of 46.13% compared to unoptimized versions. Compared to existing open-source implementations, our methods decrease latency by 64.36% to 99.1%. A final comparison with state-of-the-art element-wise quantization methods like AWQ and KVQuant shows that our VQ-LLM is practically viable, achieving latencies close or even better latencies to those at equivalent bit-widths, potentially offering greater accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02233v1">Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently hallucinate due to misaligned self-awareness, generating erroneous outputs when addressing queries beyond their knowledge boundaries. While existing approaches mitigate hallucinations via uncertainty estimation or query rejection, they suffer from computational inefficiency or sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate use of high-confidence outputs. For uncertain predictions, a slow refinement model conducts targeted reasoning to improve accuracy. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. Our work establishes a scalable paradigm for advancing LLM reliability and balancing accuracy and practical utility in error-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19134v3">Confidential Prompting: Protecting User Prompts from Cloud LLM Providers</a></div>
    <div class="paper-meta">
      📅 2025-03-04
    </div>
    <details class="paper-abstract">
      Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring model confidentiality, output invariance, and compute efficiency. We introduce Secure Partitioned Decoding (SPD), which uses confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, Prompt Obfuscation (PO), to ensure robustness against reconstruction attacks on SPD. We demonstrate our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution enables privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03257v3">Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      This paper conducts a longitudinal study over eleven months to address the limitations of prior research on the Open Ko-LLM Leaderboard, which have relied on empirical studies with restricted observation periods of only five months. By extending the analysis duration, we aim to provide a more comprehensive understanding of the progression in developing Korean large language models (LLMs). Our study is guided by three primary research questions: (1) What are the specific challenges in improving LLM performance across diverse tasks on the Open Ko-LLM Leaderboard over time? (2) How does model size impact task performance correlations across various benchmarks? (3) How have the patterns in leaderboard rankings shifted over time on the Open Ko-LLM Leaderboard?. By analyzing 1,769 models over this period, our research offers a comprehensive examination of the ongoing advancements in LLMs and the evolving nature of evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12445v3">Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Evaluation for Korean LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      The Open Ko-LLM Leaderboard has been instrumental in benchmarking Korean Large Language Models (LLMs), yet it has certain limitations. Notably, the disconnect between quantitative improvements on the overly academic leaderboard benchmarks and the qualitative impact of the models should be addressed. Furthermore, the benchmark suite is largely composed of translated versions of their English counterparts, which may not fully capture the intricacies of the Korean language. To address these issues, we propose Open Ko-LLM Leaderboard2, an improved version of the earlier Open Ko-LLM Leaderboard. The original benchmarks are entirely replaced with new tasks that are more closely aligned with real-world capabilities. Additionally, four new native Korean benchmarks are introduced to better reflect the distinct characteristics of the Korean language. Through these refinements, Open Ko-LLM Leaderboard2 seeks to provide a more meaningful evaluation for advancing Korean LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11242v3">Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 Published at ICLR 2025 (Oral)
    </div>
    <details class="paper-abstract">
      LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce Trust-Score, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by Trust-Score. Consequently, we propose Trust-Align, a method to align LLMs for improved Trust-Score performance. 26 out of 27 models aligned using Trust-Align substantially outperform competitive baselines on ASQA, QAMPARI, and ELI5. Specifically, in LLaMA-3-8b, Trust-Align outperforms FRONT on ASQA (up 12.56), QAMPARI (up 36.04), and ELI5 (up 17.69). Trust-Align also significantly enhances models' ability to correctly refuse and provide quality citations. We also demonstrate the effectiveness of Trust-Align across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at https://github.com/declare-lab/trust-align.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10038v2">POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 AAAI 25
    </div>
    <details class="paper-abstract">
      POI representation learning plays a crucial role in handling tasks related to user mobility data. Recent studies have shown that enriching POI representations with multimodal information can significantly enhance their task performance. Previously, the textual information incorporated into POI representations typically involved only POI categories or check-in content, leading to relatively weak textual features in existing methods. In contrast, large language models (LLMs) trained on extensive text data have been found to possess rich textual knowledge. However leveraging such knowledge to enhance POI representation learning presents two key challenges: first, how to extract POI-related knowledge from LLMs effectively, and second, how to integrate the extracted information to enhance POI representations. To address these challenges, we propose POI-Enhancer, a portable framework that leverages LLMs to improve POI representations produced by classic POI learning models. We first design three specialized prompts to extract semantic information from LLMs efficiently. Then, the Dual Feature Alignment module enhances the quality of the extracted information, while the Semantic Feature Fusion module preserves its integrity. The Cross Attention Fusion module then fully adaptively integrates such high-quality information into POI representations and Multi-View Contrastive Learning further injects human-understandable semantic information into these representations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our framework, showing significant improvements across all baseline representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04412v2">Spread Preference Annotation: Direct Preference Judgment for Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-04
      | 💬 ICLR 2025 Oral Presentation, 22 pages
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences becomes a key component to obtaining state-of-the-art performance, but it yields a huge cost to construct a large human-annotated preference dataset. To tackle this problem, we propose a new framework, Spread Preference Annotation with direct preference judgment (SPA), that boosts the alignment of LLMs using only a very small amount of human-annotated preference data. Our key idea is leveraging the human prior knowledge within the small (seed) data and progressively improving the alignment of LLM, by iteratively generating the responses and learning from them with the self-annotated preference data. To be specific, we propose to derive the preference label from the logits of LLM to explicitly extract the model's inherent preference. Compared to the previous approaches using external reward models or implicit in-context learning, we observe that the proposed approach is significantly more effective. In addition, we introduce a noise-aware preference learning algorithm to mitigate the risk of low quality within generated preference data. Our experimental results demonstrate that the proposed framework significantly boosts the alignment of LLMs. For example, we achieve superior alignment performance on AlpacaEval 2.0 with only 3.3% of the ground-truth preference labels in the Ultrafeedback data compared to the cases using the entire data or state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17710v4">Optimization-based Prompt Injection Attack to LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 To appear in the Proceedings of The ACM Conference on Computer and Communications Security (CCS), 2024
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12952v2">Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited significant potential in performing diverse tasks, including the ability to call functions or use external tools to enhance their performance. While current research on function calling by LLMs primarily focuses on single-turn interactions, this paper addresses the overlooked necessity for LLMs to engage in multi-turn function calling--critical for handling compositional, real-world queries that require planning with functions but not only use functions. To facilitate this, we introduce an approach, BUTTON, which generates synthetic compositional instruction tuning data via bottom-up instruction construction and top-down trajectory generation. In the bottom-up phase, we generate simple atomic tasks based on real-world scenarios and build compositional tasks using heuristic strategies based on atomic tasks. Corresponding function definitions are then synthesized for these compositional tasks. The top-down phase features a multi-agent environment where interactions among simulated humans, assistants, and tools are utilized to gather multi-turn function calling trajectories. This approach ensures task compositionality and allows for effective function and trajectory generation by examining atomic tasks within compositional tasks. We produce a dataset BUTTONInstruct comprising 8k data points and demonstrate its effectiveness through extensive experiments across various LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15770v2">Performance Review on LLM for solving leetcode problems</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive performance evaluation of Large Language Models (LLMs) in solving programming challenges from Leetcode, a widely used platform for algorithm practice and technical interviews. We began by crawling the Leetcode website to collect a diverse set of problems encompassing various difficulty levels and topics. Using this dataset, we generated solutions with multiple LLMs, including GPT-4 and GPT-3.5-turbo (ChatGPT-turbo). The generated solutions were systematically evaluated for correctness and efficiency. We employed the pass@k metric to assess the success rates within a given number of attempts and analyzed the runtime performance of the solutions. Our results highlight the strengths and limitations of current LLMs [10] in code generation and problem-solving tasks, providing insights into their potential applications and areas for improvement in automated programming assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05864v4">From Tokens to Words: On the Inner Lexicon of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Accepted to the International Conference on Learning Representations (ICLR) 2025
    </div>
    <details class="paper-abstract">
      Natural language is composed of words, but modern large language models (LLMs) process sub-words as input. A natural question raised by this discrepancy is whether LLMs encode words internally, and if so how. We present evidence that LLMs engage in an intrinsic detokenization process, where sub-word sequences are combined into coherent whole-word representations at their last token. Our experiments show that this process primarily takes place within the early and middle layers of the model. We further demonstrate its robustness to arbitrary splits (e.g., "cats" to "ca" and "ts"), typos, and importantly-to out-of-vocabulary words: when feeding the last token internal representations of such words to the model as input, it can "understand" them as the complete word despite never seeing such representations as input during training. Our findings suggest that LLMs maintain a latent vocabulary beyond the tokenizer's scope. These insights provide a practical, finetuning-free application for expanding the vocabulary of pre-trained models. By enabling the addition of new vocabulary words, we reduce input length and inference iterations, which reduces both space and model latency, with little to no loss in model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19735v2">R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Despite recent breakthroughs in reasoning-enhanced large language models (LLMs) like DeepSeek-R1, incorporating inference-time reasoning into machine translation (MT), where human translators naturally employ structured, multi-layered reasoning chain-of-thoughts (CoTs), is yet underexplored. Existing methods either design a fixed CoT tailored for a specific MT sub-task (e.g., literature translation), or rely on synthesizing CoTs unaligned with humans, limiting their adaptability to diverse translation scenarios. This paper introduces R1-Translator (R1-T1), a novel framework to achieve inference-time reasoning for general MT via reinforcement learning (RL) with human-aligned CoTs comprising six common patterns. Our approach pioneers three innovations: (1) extending reasoning-based translation beyond MT sub-tasks to six languages and diverse tasks (e.g., legal/medical domain adaptation, idiom resolution); (2) formalizing six expert-curated CoT templates that mirror hybrid human strategies like context-aware paraphrasing and back translation; and (3) enabling self-evolving CoT discovery through RL. Experimental results indicate a steady translation performance improvement in 11 languages and 40 translation directions on Flores-101 test set, especially on the languages unseen from training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15823v3">InductionBench: LLMs Fail in the Simplest Complexity Class</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 24 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable improvements in reasoning and many existing benchmarks have been addressed by models such as o1 and o3 either fully or partially. However, a majority of these benchmarks emphasize deductive reasoning, including mathematical and coding tasks in which rules such as mathematical axioms or programming syntax are clearly defined, based on which LLMs can plan and apply these rules to arrive at a solution. In contrast, inductive reasoning, where one infers the underlying rules from observed data, remains less explored. Such inductive processes lie at the heart of scientific discovery, as they enable researchers to extract general principles from empirical observations. To assess whether LLMs possess this capacity, we introduce InductionBench, a new benchmark designed to evaluate the inductive reasoning ability of LLMs. Our experimental findings reveal that even the most advanced models available struggle to master the simplest complexity classes within the subregular hierarchy of functions, highlighting a notable deficiency in current LLMs' inductive reasoning capabilities. Coda and data are available https://github.com/Wenyueh/inductive_reasoning_benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02099v1">LLMs as Educational Analysts: Transforming Multimodal Data Traces into Actionable Reading Assessment Reports</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 15 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Reading assessments are essential for enhancing students' comprehension, yet many EdTech applications focus mainly on outcome-based metrics, providing limited insights into student behavior and cognition. This study investigates the use of multimodal data sources -- including eye-tracking data, learning outcomes, assessment content, and teaching standards -- to derive meaningful reading insights. We employ unsupervised learning techniques to identify distinct reading behavior patterns, and then a large language model (LLM) synthesizes the derived information into actionable reports for educators, streamlining the interpretation process. LLM experts and human educators evaluate these reports for clarity, accuracy, relevance, and pedagogical usefulness. Our findings indicate that LLMs can effectively function as educational analysts, turning diverse data into teacher-friendly insights that are well-received by educators. While promising for automating insight generation, human oversight remains crucial to ensure reliability and fairness. This research advances human-centered AI in education, connecting data-driven analytics with practical classroom applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02067v1">AI persuading AI vs AI persuading Humans: LLMs' Differential Effectiveness in Promoting Pro-Environmental Behavior</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 17 pages, 13 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Pro-environmental behavior (PEB) is vital to combat climate change, yet turning awareness into intention and action remains elusive. We explore large language models (LLMs) as tools to promote PEB, comparing their impact across 3,200 participants: real humans (n=1,200), simulated humans based on actual participant data (n=1,200), and fully synthetic personas (n=1,200). All three participant groups faced personalized or standard chatbots, or static statements, employing four persuasion strategies (moral foundations, future self-continuity, action orientation, or "freestyle" chosen by the LLM). Results reveal a "synthetic persuasion paradox": synthetic and simulated agents significantly affect their post-intervention PEB stance, while human responses barely shift. Simulated participants better approximate human trends but still overestimate effects. This disconnect underscores LLM's potential for pre-evaluating PEB interventions but warns of its limits in predicting real-world behavior. We call for refined synthetic modeling and sustained and extended human trials to align conversational AI's promise with tangible sustainability outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02038v1">Persuasion at Play: Understanding Misinformation Dynamics in Demographic-Aware Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Existing challenges in misinformation exposure and susceptibility vary across demographic groups, as some populations are more vulnerable to misinformation than others. Large language models (LLMs) introduce new dimensions to these challenges through their ability to generate persuasive content at scale and reinforcing existing biases. This study investigates the bidirectional persuasion dynamics between LLMs and humans when exposed to misinformative content. We analyze human-to-LLM influence using human-stance datasets and assess LLM-to-human influence by generating LLM-based persuasive arguments. Additionally, we use a multi-agent LLM framework to analyze the spread of misinformation under persuasion among demographic-oriented LLM agents. Our findings show that demographic factors influence susceptibility to misinformation in LLMs, closely reflecting the demographic-based patterns seen in human susceptibility. We also find that, similar to human demographic groups, multi-agent LLMs exhibit echo chamber behavior. This research explores the interplay between humans and LLMs, highlighting demographic differences in the context of misinformation and offering insights for future interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02825v4">Randomly Sampled Language Reasoning Problems Reveal Limits of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 8 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Can LLMs pick up language structure from examples? Evidence in prior work seems to indicate yes, as pretrained models repeatedly demonstrate the ability to adapt to new language structures and vocabularies. However, this line of research typically considers languages that are present within common pretraining datasets, or otherwise share notable similarities with these seen languages. In contrast, in this work we attempt to measure models' language understanding capacity while circumventing the risk of dataset recall. We parameterize large families of language tasks recognized by deterministic finite automata (DFAs), and can thus sample novel language reasoning problems to fairly evaulate LLMs regardless of training data. We find that, even in the strikingly simple setting of 3-state DFAs, LLMs underperform unparameterized ngram models on both language recognition and synthesis tasks. These results suggest that LLMs struggle to match the ability of basic language models in recognizing and reasoning over languages that are sufficiently distinct from the ones they see at training time, underscoring the distinction between learning individual languages and possessing a general theory of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07810v4">Transformer Block Coupling and its Correlation with Generalization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Published as a conference paper at the International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language processing, and a precise understanding of the internal mechanisms driving their success is essential. In this work, we analyze the trajectories of token embeddings as they pass through transformer blocks, linearizing the system along these trajectories through their Jacobian matrices. By examining the relationships between these block Jacobians, we uncover the phenomenon of \textbf{transformer block coupling} in a multitude of LLMs, characterized by the coupling of their top singular vectors across tokens and depth. Our findings reveal that coupling \textit{positively correlates} with model performance, and that this relationship is stronger than with other hyperparameters such as parameter count, model depth, and embedding dimension. We further investigate the emergence of these properties through training, observing the progressive development of coupling, as well as increased linearity and layer-wise exponential growth in the token trajectories. Additionally, experiments with ViTs further validate emergence of coupling and its correlation between coupling and generalization, complementing our findings in LLMs. Collectively, these insights provide a novel perspective on token interactions in transformers and open directions for studying and improving training and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02016v1">Mind the (Belief) Gap: Group Identity in the World of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Social biases and belief-driven behaviors can significantly impact Large Language Models (LLMs) decisions on several tasks. As LLMs are increasingly used in multi-agent systems for societal simulations, their ability to model fundamental group psychological characteristics remains critical yet under-explored. In this study, we present a multi-agent framework that simulates belief congruence, a classical group psychology theory that plays a crucial role in shaping societal interactions and preferences. Our findings reveal that LLMs exhibit amplified belief congruence compared to humans, across diverse contexts. We further investigate the implications of this behavior on two downstream tasks: (1) misinformation dissemination and (2) LLM learning, finding that belief congruence in LLMs increases misinformation dissemination and impedes learning. To mitigate these negative impacts, we propose strategies inspired by: (1) contact hypothesis, (2) accuracy nudges, and (3) global citizenship framework. Our results show that the best strategies reduce misinformation dissemination by up to 37% and enhance learning by 11%. Bridging social psychology and AI, our work provides insights to navigate real-world interactions using LLMs while addressing belief-driven biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05218v3">Density estimation with LLMs: a geometric investigation of in-context learning trajectories</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable emergent abilities to perform in-context learning across various tasks, including time series forecasting. This work investigates LLMs' ability to estimate probability density functions (PDFs) from data observed in-context; such density estimation (DE) is a fundamental task underlying many probabilistic modeling problems. We leverage the Intensive Principal Component Analysis (InPCA) to visualize and analyze the in-context learning dynamics of LLaMA-2 models. Our main finding is that these LLMs all follow similar learning trajectories in a low-dimensional InPCA space, which are distinct from those of traditional density estimation methods like histograms and Gaussian kernel density estimation (KDE). We interpret the LLaMA in-context DE process as a KDE with an adaptive kernel width and shape. This custom kernel model captures a significant portion of LLaMA's behavior despite having only two parameters. We further speculate on why LLaMA's kernel width and shape differs from classical algorithms, providing insights into the mechanism of in-context probabilistic reasoning in LLMs. Our codebase, along with a 3D visualization of an LLM's in-context learning trajectory, is publicly available at https://github.com/AntonioLiu97/LLMICL_inPCA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01836v1">CrowdSelect: Synthetic Instruction Data Selection with Multi-LLM Wisdom</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Distilling advanced Large Language Models' instruction-following capabilities into smaller models using a selected subset has become a mainstream approach in model training. While existing synthetic instruction data selection strategies rely mainly on single-dimensional signals (i.e., reward scores, model perplexity), they fail to capture the complexity of instruction-following across diverse fields. Therefore, we investigate more diverse signals to capture comprehensive instruction-response pair characteristics and propose three foundational metrics that leverage Multi-LLM wisdom, informed by (1) diverse LLM responses and (2) reward model assessment. Building upon base metrics, we propose CrowdSelect, an integrated metric incorporating a clustering-based approach to maintain response diversity. Our comprehensive experiments demonstrate that our foundation metrics consistently improve performance across 4 base models on MT-bench and Arena-Hard. CrowdSelect, efficiently incorporating all metrics, achieves state-of-the-art performance in both Full and LoRA fine-tuning, showing improvements of 4.81% on Arena-Hard and 11.1% on MT-bench with Llama-3.2-3b-instruct. We hope our findings will bring valuable insights for future research in this direction. Code are available at https://github.com/listentm/crowdselect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01830v1">From Language to Cognition: How LLMs Outgrow the Human Language Network</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable similarity to neural activity in the human language network. However, the key properties of language shaping brain-like representations, and their evolution during training as a function of different tasks remain unclear. We here benchmark 34 training checkpoints spanning 300B tokens across 8 different model sizes to analyze how brain alignment relates to linguistic competence. Specifically, we find that brain alignment tracks the development of formal linguistic competence -- i.e., knowledge of linguistic rules -- more closely than functional linguistic competence. While functional competence, which involves world knowledge and reasoning, continues to develop throughout training, its relationship with brain alignment is weaker, suggesting that the human language network primarily encodes formal linguistic structure rather than broader cognitive functions. We further show that model size is not a reliable predictor of brain alignment when controlling for feature size and find that the correlation between next-word prediction, behavioral alignment and brain alignment fades once models surpass human language proficiency. Finally, using the largest set of rigorous neural language benchmarks to date, we show that language brain alignment benchmarks remain unsaturated, highlighting opportunities for improving future models. Taken together, our findings suggest that the human language network is best modeled by formal, rather than functional, aspects of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01821v1">On the Power of Context-Enhanced Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 76 pages, 17 figures; Pre-print
    </div>
    <details class="paper-abstract">
      We formalize a new concept for LLMs, context-enhanced learning. It involves standard gradient-based learning on text except that the context is enhanced with additional data on which no auto-regressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works. Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01768v1">SHADE-AD: An LLM-Based Framework for Synthesizing Activity Data of Alzheimer's Patients</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 7 pages, 6 figures, ACM SenSys'25
    </div>
    <details class="paper-abstract">
      Alzheimer's Disease (AD) has become an increasingly critical global health concern, which necessitates effective monitoring solutions in smart health applications. However, the development of such solutions is significantly hindered by the scarcity of AD-specific activity datasets. To address this challenge, we propose SHADE-AD, a Large Language Model (LLM) framework for Synthesizing Human Activity Datasets Embedded with AD features. Leveraging both public datasets and our own collected data from 99 AD patients, SHADE-AD synthesizes human activity videos that specifically represent AD-related behaviors. By employing a three-stage training mechanism, it broadens the range of activities beyond those collected from limited deployment settings. We conducted comprehensive evaluations of the generated dataset, demonstrating significant improvements in downstream tasks such as Human Activity Recognition (HAR) detection, with enhancements of up to 79.69%. Detailed motion metrics between real and synthetic data show strong alignment, validating the realism and utility of the synthesized dataset. These results underscore SHADE-AD's potential to advance smart health applications by providing a cost-effective, privacy-preserving solution for AD monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01767v1">Designing VR Simulation System for Clinical Communication Training with LLMs-Based Embodied Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      VR simulation in Health Professions (HP) education demonstrates huge potential, but fixed learning content with little customization limits its application beyond lab environments. To address these limitations in the context of VR for patient communication training, we conducted a user-centered study involving semi-structured interviews with advanced HP students to understand their challenges in clinical communication training and perceptions of VR-based solutions. From this, we derived design insights emphasizing the importance of realistic scenarios, simple interactions, and unpredictable dialogues. Building on these insights, we developed the Virtual AI Patient Simulator (VAPS), a novel VR system powered by Large Language Models (LLMs) and Embodied Conversational Agents (ECAs), supporting dynamic and customizable patient interactions for immersive learning. We also provided an example of how clinical professors could use user-friendly design forms to create personalized scenarios that align with course objectives in VAPS and discuss future implications of integrating AI-driven technologies into VR education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01714v1">Word Form Matters: LLMs' Semantic Reconstruction under Typoglycemia</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 14 pages, 10 figures, submitted to ACL Rolling Review, February 2025 cycle, see https://github.com/Aurora-cx/TypoLLM
    </div>
    <details class="paper-abstract">
      Human readers can efficiently comprehend scrambled words, a phenomenon known as Typoglycemia, primarily by relying on word form; if word form alone is insufficient, they further utilize contextual cues for interpretation. While advanced large language models (LLMs) exhibit similar abilities, the underlying mechanisms remain unclear. To investigate this, we conduct controlled experiments to analyze the roles of word form and contextual information in semantic reconstruction and examine LLM attention patterns. Specifically, we first propose SemRecScore, a reliable metric to quantify the degree of semantic reconstruction, and validate its effectiveness. Using this metric, we study how word form and contextual information influence LLMs' semantic reconstruction ability, identifying word form as the core factor in this process. Furthermore, we analyze how LLMs utilize word form and find that they rely on specialized attention heads to extract and process word form information, with this mechanism remaining stable across varying levels of word scrambling. This distinction between LLMs' fixed attention patterns primarily focused on word form and human readers' adaptive strategy in balancing word form and contextual information provides insights into enhancing LLM performance by incorporating human-like, context-aware mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01710v1">Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have driven significant progress in zero-shot text-to-speech (TTS) synthesis. However, existing foundation models rely on multi-stage processing or complex architectures for predicting multiple codebooks, limiting efficiency and integration flexibility. To overcome these challenges, we introduce Spark-TTS, a novel system powered by BiCodec, a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes. This disentangled representation, combined with the Qwen2.5 LLM and a chain-of-thought (CoT) generation approach, enables both coarse-grained control (e.g., gender, speaking style) and fine-grained adjustments (e.g., precise pitch values, speaking rate). To facilitate research in controllable TTS, we introduce VoxBox, a meticulously curated 100,000-hour dataset with comprehensive attribute annotations. Extensive experiments demonstrate that Spark-TTS not only achieves state-of-the-art zero-shot voice cloning but also generates highly customizable voices that surpass the limitations of reference-based synthesis. Source code, pre-trained models, and audio samples are available at https://github.com/SparkAudio/Spark-TTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01704v1">DILEMMA: Joint LLM Quantization and Distributed LLM Inference Over Edge Computing Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      With a recent trend of using Large Language Models (LLMs) for different applications within smart cities, there is a need for pushing these models toward the edge of network while still preserving their performance. Edge Computing (EC) as a physically closer computing resource to the end users can help to reduce the communication delay for serving end users' tasks for LLM-dependent services. However, EC servers have limited capacity in terms of communication, computation, and storage capacity. This paper introduces DILEMMA, a novel framework addressing the challenges of deploying LLMs in EC systems by jointly optimizing layer placement and layer quantization in EC systems. DILEMMA formulates an Integer Linear Programming problem to minimize total inference delay while ensuring acceptable LLM performance levels, leveraging layer-wise quantization and knowledge distillation for LLM performance control. Experimental evaluations on OPT-350 model using the SQuAD dataset demonstrate that DILEMMA achieves a quantization ratio of up to 12.75% while preserving model loss, highlighting its effectiveness in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01694v1">Student engagement in collaborative learning with AI agents in an LLM-empowered learning environment: A cluster analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Integrating LLM models into educational practice fosters personalized learning by accommodating the diverse behavioral patterns of different learner types. This study aims to explore these learner types within a novel interactive setting, providing a detailed analysis of their distinctive characteristics and interaction dynamics. The research involved 110 students from a university in China, who engaged with multiple LLM agents in an LLM-empowered learning environment, completing coursework across six modules. Data on the students' non-cognitive traits, course engagement, and AI interaction patterns were collected and analyzed. Using hierarchical cluster analysis, the students were classified into three distinct groups: active questioners, responsive navigators, and silent listeners. Epistemic network analysis was then applied to further delineate the interaction profiles and cognitive engagement of different types of learners. The findings underscore how different learner types engage with human-AI interactive learning and offer practical implications for the design of adaptive educational systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01688v1">When an LLM is apprehensive about its answers -- and when its uncertainty is justified</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Uncertainty estimation is crucial for evaluating Large Language Models (LLMs), particularly in high-stakes domains where incorrect answers result in significant consequences. Numerous approaches consider this problem, while focusing on a specific type of uncertainty, ignoring others. We investigate what estimates, specifically token-wise entropy and model-as-judge (MASJ), would work for multiple-choice question-answering tasks for different question topics. Our experiments consider three LLMs: Phi-4, Mistral, and Qwen of different sizes from 1.5B to 72B and $14$ topics. While MASJ performs similarly to a random error predictor, the response entropy predicts model error in knowledge-dependent domains and serves as an effective indicator of question difficulty: for biology ROC AUC is $0.73$. This correlation vanishes for the reasoning-dependent domain: for math questions ROC-AUC is $0.55$. More principally, we found out that the entropy measure required a reasoning amount. Thus, data-uncertainty related entropy should be integrated within uncertainty estimates frameworks, while MASJ requires refinement. Moreover, existing MMLU-Pro samples are biased, and should balance required amount of reasoning for different subdomains to provide a more fair assessment of LLMs performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01670v1">Evaluating LLMs' Assessment of Mixed-Context Hallucination Through the Lens of Summarization</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 8 pages, 5 figures for main body
    </div>
    <details class="paper-abstract">
      With the rapid development of large language models (LLMs), LLM-as-a-judge has emerged as a widely adopted approach for text quality evaluation, including hallucination evaluation. While previous studies have focused exclusively on single-context evaluation (e.g., discourse faithfulness or world factuality), real-world hallucinations typically involve mixed contexts, which remains inadequately evaluated. In this study, we use summarization as a representative task to comprehensively evaluate LLMs' capability in detecting mixed-context hallucinations, specifically distinguishing between factual and non-factual hallucinations. Through extensive experiments across direct generation and retrieval-based models of varying scales, our main observations are: (1) LLMs' intrinsic knowledge introduces inherent biases in hallucination evaluation; (2) These biases particularly impact the detection of factual hallucinations, yielding a significant performance bottleneck; (3) The fundamental challenge lies in effective knowledge utilization, balancing between LLMs' intrinsic knowledge and external context for accurate mixed-context hallucination evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01658v1">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 13pages, 4 figures, 6tables
    </div>
    <details class="paper-abstract">
      Personalizing large language models (LLMs) is important for aligning outputs with diverse user preferences, yet existing methods struggle with flexibility and generalization. We propose CoPL (Collaborative Preference Learning), a graph-based collaborative filtering framework that models user-response relationships to enhance preference estimation, particularly in sparse annotation settings. By integrating a mixture of LoRA experts, CoPL efficiently fine-tunes LLMs while dynamically balancing shared and user-specific preferences. Additionally, an optimization-free adaptation strategy enables generalization to unseen users without fine-tuning. Experiments on UltraFeedback-P demonstrate that CoPL outperforms existing personalized reward models, effectively capturing both common and controversial preferences, making it a scalable solution for personalized LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01642v1">Graph-Augmented Reasoning: Evolving Step-by-Step Knowledge Graph Retrieval for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Recent large language model (LLM) reasoning, despite its success, suffers from limited domain knowledge, susceptibility to hallucinations, and constrained reasoning depth, particularly in small-scale models deployed in resource-constrained environments. This paper presents the first investigation into integrating step-wise knowledge graph retrieval with step-wise reasoning to address these challenges, introducing a novel paradigm termed as graph-augmented reasoning. Our goal is to enable frozen, small-scale LLMs to retrieve and process relevant mathematical knowledge in a step-wise manner, enhancing their problem-solving abilities without additional training. To this end, we propose KG-RAR, a framework centered on process-oriented knowledge graph construction, a hierarchical retrieval strategy, and a universal post-retrieval processing and reward model (PRP-RM) that refines retrieved information and evaluates each reasoning step. Experiments on the Math500 and GSM8K benchmarks across six models demonstrate that KG-RAR yields encouraging results, achieving a 20.73\% relative improvement with Llama-3B on Math500.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01631v1">No Evidence for LLMs Being Useful in Problem Reframing</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 14 pages, 10 figures, 2 tables, Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Problem reframing is a designerly activity wherein alternative perspectives are created to recast what a stated design problem is about. Generating alternative problem frames is challenging because it requires devising novel and useful perspectives that fit the given problem context. Large language models (LLMs) could assist this activity via their generative capability. However, it is not clear whether they can help designers produce high-quality frames. Therefore, we asked if there are benefits to working with LLMs. To this end, we compared three ways of using LLMs (N=280): 1) free-form, 2) direct generation, and 3) a structured approach informed by a theory of reframing. We found that using LLMs does not help improve the quality of problem frames. In fact, it increases the competence gap between experienced and inexperienced designers. Also, inexperienced ones perceived lower agency when working with LLMs. We conclude that there is no benefit to using LLMs in problem reframing and discuss possible factors for this lack of effect.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01550v1">None of the Above, Less of the Right: Parallel Patterns between Humans and LLMs on Multi-Choice Questions Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Multiple-choice exam questions with "None of the above" (NA) options have been extensively studied in educational testing, in which existing research suggests that they better assess true knowledge. However, their impact on Large Language Models (LLMs) evaluation remains underexplored. Through systematic experiments with 28 LLMs on the MMLU benchmark, we examine how NA options affect model performance and confidence calibration. Our analysis reveals that NA options, when used as the correct answer, lead to a consistent 30-50\% performance drop across models regardless of scale--suggesting that LLMs lack the meta-cognitive ability to systematically evaluate and reject all given options when none are correct. This degradation shows strong domain dependence, with minimal impact on mathematical reasoning (14.6\% drop) but severe effects on tasks requiring uncertainty handling like business ethics (48.1\% drop). Our results highlight important implications for benchmark design and raise questions about LLMs' ability to handle uncertainty in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01539v1">Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 12 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, and DeepSeek-v2.5 in identifying implicit toxic language, compared to both direct prompting and Chain-of-Thought. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01532v1">Unmasking Implicit Bias: Evaluating Persona-Prompted LLM Responses in Power-Disparate Social Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 NAACL 2025; 10 pages of main text
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in simulating human behaviour and social intelligence. However, they risk perpetuating societal biases, especially when demographic information is involved. We introduce a novel framework using cosine distance to measure semantic shifts in responses and an LLM-judged Preference Win Rate (WR) to assess how demographic prompts affect response quality across power-disparate social scenarios. Evaluating five LLMs over 100 diverse social scenarios and nine demographic axes, our findings suggest a "default persona" bias toward middle-aged, able-bodied, native-born, Caucasian, atheistic males with centrist views. Moreover, interactions involving specific demographics are associated with lower-quality responses. Lastly, the presence of power disparities increases variability in response semantics and quality across demographic groups, suggesting that implicit biases may be heightened under power-imbalanced conditions. These insights expose the demographic biases inherent in LLMs and offer potential paths toward future bias mitigation efforts in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01513v1">Evaluation and Facilitation of Online Discussions in the LLM Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      We present a survey of methods for assessing and enhancing the quality of online discussions, focusing on the potential of Large Language Models (LLMs). While online discourses aim, at least in theory, to foster mutual understanding, they often devolve into harmful exchanges, such as hate speech, threatening social cohesion and democratic values. Recent advancements in LLMs enable facilitation agents that not only moderate content, but also actively improve the quality of interactions. Our survey synthesizes ideas from Natural Language Processing (NLP) and Social Sciences to provide (a) a new taxonomy on discussion quality evaluation, (b) an overview of intervention and facilitation strategies, along with a new taxonomy on conversation facilitation datasets, (c) an LLM-oriented roadmap of good practices and future research directions, from technological and societal perspectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01940v1">AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in tool learning. In real-world scenarios, user queries are often ambiguous and incomplete, requiring effective clarification. However, existing interactive clarification approaches face two critical limitations: reliance on manually constructed datasets and lack of error correction mechanisms during multi-turn clarification. We present AskToAct, which addresses these challenges by exploiting the structural mapping between queries and their tool invocation solutions. Our key insight is that tool parameters naturally represent explicit user intents. By systematically removing key parameters from queries while retaining them as ground truth, we enable automated construction of high-quality training data. We further enhance model robustness by fine-tuning on error-correction augmented data using selective masking mechanism, enabling dynamic error detection during clarification interactions. Comprehensive experiments demonstrate that AskToAct significantly outperforms existing approaches, achieving above 79% accuracy in recovering critical unspecified intents and enhancing clarification efficiency by an average of 48.34% while maintaining high accuracy in tool invocation. Our framework exhibits robust performance across varying complexity levels and successfully generalizes to entirely unseen APIs without additional training, achieving performance comparable to GPT-4 with substantially fewer computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01483v1">KurTail : Kurtosis-based LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      One of the challenges of quantizing a large language model (LLM) is the presence of outliers. Outliers often make uniform quantization schemes less effective, particularly in extreme cases such as 4-bit quantization. We introduce KurTail, a new post-training quantization (PTQ) scheme that leverages Kurtosis-based rotation to mitigate outliers in the activations of LLMs. Our method optimizes Kurtosis as a measure of tailedness. This approach enables the quantization of weights, activations, and the KV cache in 4 bits. We utilize layer-wise optimization, ensuring memory efficiency. KurTail outperforms existing quantization methods, offering a 13.3\% boost in MMLU accuracy and a 15.5\% drop in Wiki perplexity compared to QuaRot. It also outperforms SpinQuant with a 2.6\% MMLU gain and reduces perplexity by 2.9\%, all while reducing the training cost. For comparison, learning the rotation using SpinQuant for Llama3-70B requires at least four NVIDIA H100 80GB GPUs, whereas our method requires only a single GPU, making it a more accessible solution for consumer GPU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01442v1">Leveraging LLMs for Mental Health: Detection and Recommendations from Social Discussions</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 5 pages, 4 figures, 3 tables, to be published in WI-IAT 2024
    </div>
    <details class="paper-abstract">
      Textual data from social platforms captures various aspects of mental health through discussions around and across issues, while users reach out for help and others sympathize and offer support. We propose a comprehensive framework that leverages Natural Language Processing (NLP) and Generative AI techniques to identify and assess mental health disorders, detect their severity, and create recommendations for behavior change and therapeutic interventions based on users' posts on Reddit. To classify the disorders, we use rule-based labeling methods as well as advanced pre-trained NLP models to extract nuanced semantic features from the data. We fine-tune domain-adapted and generic pre-trained NLP models based on predictions from specialized Large Language Models (LLMs) to improve classification accuracy. Our hybrid approach combines the generalization capabilities of pre-trained models with the domain-specific insights captured by LLMs, providing an improved understanding of mental health discourse. Our findings highlight the strengths and limitations of each model, offering valuable insights into their practical applicability. This research potentially facilitates early detection and personalized care to aid practitioners and aims to facilitate timely interventions and improve overall well-being, thereby contributing to the broader field of mental health surveillance and digital health analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01319v1">ABFS: Natural Robustness Testing for LLM-based NLP Software</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Owing to the exceptional performance of Large Language Models (LLMs) in Natural Language Processing (NLP) tasks, LLM-based NLP software has rapidly gained traction across various domains, such as financial analysis and content moderation. However, these applications frequently exhibit robustness deficiencies, where slight perturbations in input (prompt+example) may lead to erroneous outputs. Current robustness testing methods face two main limitations: (1) low testing effectiveness, limiting the applicability of LLM-based software in safety-critical scenarios, and (2) insufficient naturalness of test cases, reducing the practical value of testing outcomes. To address these issues, this paper proposes ABFS, a straightforward yet effective automated testing method that, for the first time, treats the input prompts and examples as a unified whole for robustness testing. Specifically, ABFS formulates the testing process as a combinatorial optimization problem, employing Best-First Search to identify successful test cases within the perturbation space and designing a novel Adaptive control strategy to enhance test case naturalness. We evaluate the robustness testing performance of ABFS on three datasets across five threat models. On Llama2-13b, the traditional StressTest achieves only a 13.273% success rate, while ABFS attains a success rate of 98.064%, supporting a more comprehensive robustness assessment before software deployment. Compared to baseline methods, ABFS introduces fewer modifications to the original input and consistently generates test cases with superior naturalness. Furthermore, test cases generated by ABFS exhibit stronger transferability and higher testing efficiency, significantly reducing testing costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01295v1">CodeArena: A Collective Evaluation Platform for LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have reshaped code generation by synergizing their exceptional comprehension of natural language and programming syntax, thereby substantially boosting developer productivity. These advancements have prompted numerous efforts to quantitatively evaluate their coding capabilities. However, persistent challenges, such as benchmark leakage, data dissipation, and limited system accessibility, continue to impede a timely and accurate assessment. To address these limitations, we introduce CodeArena, an online evaluation framework tailored for LLM code generation. The key innovation is a collective evaluation mechanism, which dynamically recalibrates individual model scores based on the holistic performance of all participating models, mitigating score biases caused by widespread benchmark leakage. In addition, CodeArena ensures open access to all submitted solutions and test cases and provides automation-friendly APIs to streamline the code evaluation workflow. Our main contributions are: (1) a collective evaluation system for unbiased assessment, (2) a public repository of solutions and test cases, and (3) automation-ready APIs for seamless integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01236v1">LLM-Advisor: An LLM Benchmark for Cost-efficient Path Planning across Multiple Terrains</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Multi-terrain cost-efficient path planning is a crucial task in robot navigation, requiring the identification of a path from the start to the goal that not only avoids obstacles but also minimizes travel costs. This is especially crucial for real-world applications where robots need to navigate diverse terrains in outdoor environments, where recharging or refueling is difficult. However, there is very limited research on this topic. In this paper, we develop a prompt-based approach, LLM-Advisor, which leverages large language models (LLMs) as effective advisors for path planning. The LLM-Advisor selectively provides suggestions, demonstrating its ability to recognize when no modifications are necessary. When suggestions are made, 70.59% of the paths suggested for the A* algorithm, 69.47% for the RRT* algorithm, and 78.70% for the LLM-A* algorithm achieve greater cost efficiency. Since LLM-Advisor may occasionally lack common sense in their suggestions, we propose two hallucination-mitigation strategies. Furthermore, we experimentally verified that GPT-4o performs poorly in zero-shot path planning, even when terrain descriptions are clearly provided, demonstrating its low spatial awareness. We also experimentally demonstrate that using an LLM as an advisor is more effective than directly integrating it into the path-planning loop. Since LLMs may generate hallucinations, using LLMs in the loop of a search-based method (such as A*) may lead to a higher number of failed paths, demonstrating that our proposed LLM-Advisor is a better choice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01194v1">Cancer Type, Stage and Prognosis Assessment from Pathology Reports using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant promise across various natural language processing tasks. However, their application in the field of pathology, particularly for extracting meaningful insights from unstructured medical texts such as pathology reports, remains underexplored and not well quantified. In this project, we leverage state-of-the-art language models, including the GPT family, Mistral models, and the open-source Llama models, to evaluate their performance in comprehensively analyzing pathology reports. Specifically, we assess their performance in cancer type identification, AJCC stage determination, and prognosis assessment, encompassing both information extraction and higher-order reasoning tasks. Based on a detailed analysis of their performance metrics in a zero-shot setting, we developed two instruction-tuned models: Path-llama3.1-8B and Path-GPT-4o-mini-FT. These models demonstrated superior performance in zero-shot cancer type identification, staging, and prognosis assessment compared to the other models evaluated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01935v1">MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 https://github.com/MultiagentBench/MARBLE
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities as autonomous agents, yet existing benchmarks either focus on single-agent tasks or are confined to narrow domains, failing to capture the dynamics of multi-agent coordination and competition. In this paper, we introduce MultiAgentBench, a comprehensive benchmark designed to evaluate LLM-based multi-agent systems across diverse, interactive scenarios. Our framework measures not only task completion but also the quality of collaboration and competition using novel, milestone-based key performance indicators. Moreover, we evaluate various coordination protocols (including star, chain, tree, and graph topologies) and innovative strategies such as group discussion and cognitive planning. Notably, gpt-4o-mini reaches the average highest task score, graph structure performs the best among coordination protocols in the research scenario, and cognitive planning improves milestone achievement rates by 3%. Code and datasets are public available at https://github.com/MultiagentBench/MARBLE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01150v1">MiLiC-Eval: Benchmarking Multilingual LLMs for China's Minority Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Code and data available at https://github.com/luciusssss/MiLiC-Eval
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in high-resource languages but struggle with low-resource languages (LRLs), particularly those spoken by minority communities in China, such as Tibetan, Uyghur, Kazakh, and Mongolian. To systematically track the progress in these languages, we introduce MiLiC-Eval, a benchmark designed for minority languages in China, featuring 24K instances across 9 tasks. MiLiC-Eval focuses on underrepresented writing systems and provides a fine-grained assessment of linguistic and problem-solving skills. Our evaluation reveals that LLMs perform poorly on syntax-intensive tasks and multi-script languages. We further demonstrate how MiLiC-Eval can help advance LRL research in handling diverse writing systems and understanding the process of language adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01141v1">How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01131v1">Beyond QA Pairs: Assessing Parameter-Efficient Fine-Tuning for Fact Embedding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Presented at the Workshop on Preparing Good Data for Generative AI: Challenges and Approaches (Good-Data) in conjunction with AAAI 2025. The authors retain the copyright
    </div>
    <details class="paper-abstract">
      This paper presents an extensive examination of Parameter-Efficient Fine-Tuning (PEFT) for embedding domain specific facts into Large Language Models (LLMs), focusing on improving the fine-tuning process by categorizing question-answer (QA) pairs into Factual and Conceptual classes using a BERT-based classifier. Two distinct Llama-2 models are fine-tuned based on these classifications and evaluated using larger models like GPT-3.5 Turbo and Gemini. Our results indicate that models trained on conceptual datasets outperform those trained on factual datasets. Additionally, we compare the efficiency of two synthetic fine-tuning dataset generation techniques, D-RAG and D-Naive, with D-Naive demonstrating superior performance. Although PEFT has shown effectiveness, our research indicates that it may not be the most optimal method for embedding facts into LLMs. However, it has demonstrated exceptional performance in instruction-based tasks. Our findings are reinforced by a 1000-sample dataset in the data center domain, where the fine-tuned Llama-2 7B model significantly outperforms the baseline model in generating product recommendations. Our study highlights the importance of QA pair categorization and synthetic dataset generation techniques in enhancing the performance of LLMs in specific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01090v1">Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Knowledge editing aims to update outdated information in Large Language Models (LLMs). A representative line of study is locate-then-edit methods, which typically employ causal tracing to identify the modules responsible for recalling factual knowledge about entities. However, we find these methods are often sensitive only to changes in the subject entity, leaving them less effective at adapting to changes in relations. This limitation results in poor editing locality, which can lead to the persistence of irrelevant or inaccurate facts, ultimately compromising the reliability of LLMs. We believe this issue arises from the insufficient precision of knowledge localization. To address this, we propose a Fine-grained Neuron-level Knowledge Editing (FiNE) method that enhances editing locality without affecting overall success rates. By precisely identifying and modifying specific neurons within feed-forward networks, FiNE significantly improves knowledge localization and editing. Quantitative experiments demonstrate that FiNE efficiently achieves better overall performance compared to existing techniques, providing new insights into the localization and modification of knowledge within LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01064v1">Scientific Reasoning: Assessment of Multimodal Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can answer questions and reason about complex tasks, also from the scientific domain. We assess several multimodal LLMs (MLLMs) on ScienceQA and find that Gemini models show the highest accuracy with little context, and the highest textual similarity to human explanations with richer context. Adapter-tuning of smaller MLLMs did not lead to any reliable performance. Training from Gemini outputs consistently underperformed training from the original data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04671v3">CUIfy the XR: An Open-Source Package to Embed LLM-powered Conversational Agents in XR</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 7th IEEE International Conference on Artificial Intelligence & eXtended and Virtual Reality (IEEE AIxVR 2025)
    </div>
    <details class="paper-abstract">
      Recent developments in computer graphics, machine learning, and sensor technologies enable numerous opportunities for extended reality (XR) setups for everyday life, from skills training to entertainment. With large corporations offering affordable consumer-grade head-mounted displays (HMDs), XR will likely become pervasive, and HMDs will develop as personal devices like smartphones and tablets. However, having intelligent spaces and naturalistic interactions in XR is as important as technological advances so that users grow their engagement in virtual and augmented spaces. To this end, large language model (LLM)--powered non-player characters (NPCs) with speech-to-text (STT) and text-to-speech (TTS) models bring significant advantages over conventional or pre-scripted NPCs for facilitating more natural conversational user interfaces (CUIs) in XR. This paper provides the community with an open-source, customizable, extendable, and privacy-aware Unity package, CUIfy, that facilitates speech-based NPC-user interaction with widely used LLMs, STT, and TTS models. Our package also supports multiple LLM-powered NPCs per environment and minimizes latency between different computational models through streaming to achieve usable interactions between users and NPCs. We publish our source code in the following repository: https://gitlab.lrz.de/hctl/cuify
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11142v2">NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05643v3">TRACE: Temporal Grounding Video LLM via Causal Event Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG) is a crucial capability for video understanding models and plays a vital role in downstream tasks such as video browsing and editing. To effectively handle various tasks simultaneously and enable zero-shot prediction, there is a growing trend in employing video LLMs for VTG tasks. However, current video LLM-based methods rely exclusively on natural language generation, lacking the ability to model the clear structure inherent in videos, which restricts their effectiveness in tackling VTG tasks. To address this issue, this paper first formally introduces causal event modeling framework, which represents video LLM outputs as sequences of events, and predict the current event using previous events, video inputs, and textural instructions. Each event consists of three components: timestamps, salient scores, and textual captions. We then propose a novel task-interleaved video LLM called TRACE to effectively implement the causal event modeling framework in practice. The TRACE process visual frames, timestamps, salient scores, and text as distinct tasks, employing various encoders and decoding heads for each. Task tokens are arranged in an interleaved sequence according to the causal event modeling framework's formulation. Extensive experiments on various VTG tasks and datasets demonstrate the superior performance of TRACE compared to state-of-the-art video LLMs. Our model and code are available at https://github.com/gyxxyg/TRACE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11355v2">"Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Please visit https://llm-catastrophic-risks.github.io for a quick tour of our project
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07260v2">LaERC-S: Improving LLM-based Emotion Recognition in Conversation with Speaker Characteristics</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Emotion recognition in conversation (ERC), the task of discerning human emotions for each utterance within a conversation, has garnered significant attention in human-computer interaction systems. Previous ERC studies focus on speaker-specific information that predominantly stems from relationships among utterances, which lacks sufficient information around conversations. Recent research in ERC has sought to exploit pre-trained large language models (LLMs) with speaker modelling to comprehend emotional states. Although these methods have achieved encouraging results, the extracted speaker-specific information struggles to indicate emotional dynamics. In this paper, motivated by the fact that speaker characteristics play a crucial role and LLMs have rich world knowledge, we present LaERC-S, a novel framework that stimulates LLMs to explore speaker characteristics involving the mental state and behavior of interlocutors, for accurate emotion predictions. To endow LLMs with this knowledge information, we adopt the two-stage learning to make the models reason speaker characteristics and track the emotion of the speaker in complex conversation scenarios. Extensive experiments on three benchmark datasets demonstrate the superiority of LaERC-S, reaching the new state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03856v4">Q-Adapter: Customizing Pre-trained LLMs to New Preferences with Forgetting Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Camera ready version of ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), trained on a large amount of corpus, have demonstrated remarkable abilities. However, it may not be sufficient to directly apply open-source LLMs like Llama to certain real-world scenarios, since most of them are trained for \emph{general} purposes. Thus, the demands for customizing publicly available LLMs emerge, but are currently under-studied. In this work, we consider customizing pre-trained LLMs with new human preferences. Specifically, the LLM should not only meet the new preference but also preserve its original capabilities after customization. Drawing inspiration from the observation that human preference can be expressed as a reward model, we propose to cast LLM customization as optimizing the sum of two reward functions, one of which (denoted as $r_1$) was used to pre-train the LLM while the other (denoted as $r_2$) characterizes the new human preference. The obstacle here is that both reward functions are unknown, making the application of modern reinforcement learning methods infeasible. Thanks to the residual Q-learning framework, we can restore the customized LLM with the pre-trained LLM and the \emph{residual Q-function} without the reward function $r_1$. Moreover, we find that for a fixed pre-trained LLM, the reward function $r_2$ can be derived from the residual Q-function, enabling us to directly learn the residual Q-function from the new human preference data upon the Bradley-Terry model. We name our method Q-Adapter as it introduces an adapter module to approximate the residual Q-function for customizing the pre-trained LLM towards the new preference. Experiments based on the Llama-3.1 model on the DSP dataset and HH-RLHF dataset illustrate the superior effectiveness of Q-Adapter on both retaining existing knowledge and learning new preferences. Code is available at https://github.com/mansicer/Q-Adapter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14866v5">PAPILLON: Efficient and Stealthy Fuzz Testing-Powered Jailbreaks for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have excelled in various tasks but are still vulnerable to jailbreaking attacks, where attackers create jailbreak prompts to mislead the model to produce harmful or offensive content. Current jailbreak methods either rely heavily on manually crafted templates, which pose challenges in scalability and adaptability, or struggle to generate semantically coherent prompts, making them easy to detect. Additionally, most existing approaches involve lengthy prompts, leading to higher query costs. In this paper, to remedy these challenges, we introduce a novel jailbreaking attack framework called PAPILLON, which is an automated, black-box jailbreaking attack framework that adapts the black-box fuzz testing approach with a series of customized designs. Instead of relying on manually crafted templates,PAPILLON starts with an empty seed pool, removing the need to search for any related jailbreaking templates. We also develop three novel question-dependent mutation strategies using an LLM helper to generate prompts that maintain semantic coherence while significantly reducing their length. Additionally, we implement a two-level judge module to accurately detect genuine successful jailbreaks. We evaluated PAPILLON on 7 representative LLMs and compared it with 5 state-of-the-art jailbreaking attack strategies. For proprietary LLM APIs, such as GPT-3.5 turbo, GPT-4, and Gemini-Pro, PAPILLONs achieves attack success rates of over 90%, 80%, and 74%, respectively, exceeding existing baselines by more than 60\%. Additionally, PAPILLON can maintain high semantic coherence while significantly reducing the length of jailbreak prompts. When targeting GPT-4, PAPILLON can achieve over 78% attack success rate even with 100 tokens. Moreover, PAPILLON demonstrates transferability and is robust to state-of-the-art defenses. Code: https://github.com/aaFrostnova/Papillon
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02683v2">DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 Accepted into ICLR 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      As users increasingly seek guidance from LLMs for decision-making in daily life, many of these decisions are not clear-cut and depend significantly on the personal values and ethical standards of people. We present DailyDilemmas, a dataset of 1,360 moral dilemmas encountered in everyday life. Each dilemma presents two possible actions, along with affected parties and relevant human values for each action. Based on these dilemmas, we gather a repository of human values covering diverse everyday topics, such as interpersonal relationships, workplace, and environmental issues. With DailyDilemmas, we evaluate LLMs on these dilemmas to determine what action they will choose and the values represented by these action choices. Then, we analyze values through the lens of five theoretical frameworks inspired by sociology, psychology, and philosophy, including the World Values Survey, Moral Foundations Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, and Plutchik's Wheel of Emotions. For instance, we find LLMs are most aligned with self-expression over survival in World Values Survey and care over loyalty in Moral Foundations Theory. Interestingly, we find substantial preference differences in models for some core values. For example, for truthfulness, Mixtral-8x7B neglects it by 9.7% while GPT-4-turbo selects it by 9.4%. We also study the recent guidance released by OpenAI (ModelSpec), and Anthropic (Constitutional AI) to understand how their designated principles reflect their models' actual value prioritization when facing nuanced moral reasoning in daily-life settings. Finally, we find that end users cannot effectively steer such prioritization using system prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07190v2">Understanding LLMs' Fluid Intelligence Deficiency: An Analysis of the ARC Task</a></div>
    <div class="paper-meta">
      📅 2025-03-03
      | 💬 22 pages, 9 figures, accepted by NAACL 2025 main conference
    </div>
    <details class="paper-abstract">
      While LLMs have exhibited strong performance on various NLP tasks, it is noteworthy that most of these tasks rely on utilizing the vast amount of knowledge encoded in LLMs' parameters, rather than solving new problems without prior knowledge. In cognitive research, the latter ability is referred to as fluid intelligence, which is considered to be critical for assessing human intelligence. Recent research on fluid intelligence assessments has highlighted significant deficiencies in LLMs' abilities. In this paper, we analyze the challenges LLMs face in demonstrating fluid intelligence through controlled experiments, using the most representative ARC task as an example. Our study revealed three major limitations in existing LLMs: limited ability for skill composition, unfamiliarity with abstract input formats, and the intrinsic deficiency of left-to-right decoding. Our data and code can be found in https://wujunjie1998.github.io/araoc-benchmark.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02886v2">TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has driven growing demand for processing extended context sequences in contemporary applications. However, this progress faces two major challenges: performance degradation due to sequence lengths out-of-distribution, and excessively long inference times caused by the quadratic computational complexity of attention. These issues hinder the application of LLMs in long-context scenarios. In this paper, we propose Dynamic Token-Level KV Cache Selection (TokenSelect), a training-free method for efficient and accurate long-context inference. TokenSelect builds upon the observation of non-contiguous attention sparsity, using Query-Key dot products to measure per-head KV Cache criticality at token-level. By per-head soft voting mechanism, TokenSelect selectively involves a few critical KV cache tokens in attention calculation without sacrificing accuracy. To further accelerate TokenSelect, we design the Selection Cache based on observations of consecutive Query similarity and implemented efficient dot product kernel, significantly reducing the overhead. A comprehensive evaluation of TokenSelect demonstrates up to 23.84x speedup in attention computation and up to 2.28x acceleration in end-to-end latency, while providing superior performance compared to state-of-the-art long-context inference methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v2">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code is available at https://github.com/WujiangXu/AgenticMemory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00617v4">Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-03
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) has achieved great success in aligning large language models (LLMs) with human preferences. Prevalent RLHF approaches are reward-based, following the Bradley-Terry (BT) model assumption, which may not fully capture the complexity of human preferences. In this paper, we explore RLHF under a general preference framework and approach it from a game-theoretic perspective. Specifically, we formulate the problem as a two-player game and propose a novel online algorithm, iterative Nash policy optimization (INPO). The key idea is to let the policy play against itself via no-regret learning, thereby approximating the Nash policy. Unlike previous methods, INPO bypasses the need for estimating the expected win rate for individual responses, which typically incurs high computational or annotation costs. Instead, we introduce a new loss objective that is directly minimized over a preference dataset. We provide theoretical analysis for our approach and demonstrate its effectiveness through experiments on various representative benchmarks. With an LLaMA-3-8B-based SFT model, INPO achieves a 42.6% length-controlled win rate on AlpacaEval 2.0 and a 37.8% win rate on Arena-Hard, showing substantial improvement over the state-of-the-art online RLHF algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01013v1">Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Time series analysis provides essential insights for real-world system dynamics and informs downstream decision-making, yet most existing methods often overlook the rich contextual signals present in auxiliary modalities. To bridge this gap, we introduce TimeXL, a multi-modal prediction framework that integrates a prototype-based time series encoder with three collaborating Large Language Models (LLMs) to deliver more accurate predictions and interpretable explanations. First, a multi-modal prototype-based encoder processes both time series and textual inputs to generate preliminary forecasts alongside case-based rationales. These outputs then feed into a prediction LLM, which refines the forecasts by reasoning over the encoder's predictions and explanations. Next, a reflection LLM compares the predicted values against the ground truth, identifying textual inconsistencies or noise. Guided by this feedback, a refinement LLM iteratively enhances text quality and triggers encoder retraining. This closed-loop workflow -- prediction, critique (reflect), and refinement -- continuously boosts the framework's performance and interpretability. Empirical evaluations on four real-world datasets demonstrate that TimeXL achieves up to 8.9\% improvement in AUC and produces human-centric, multi-modal explanations, highlighting the power of LLM-driven reasoning for time series prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01007v1">From Vague Instructions to Task Plans: A Feedback-Driven HRC Task Planning Framework based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated their potential as planners in human-robot collaboration (HRC) scenarios, offering a promising alternative to traditional planning methods. LLMs, which can generate structured plans by reasoning over natural language inputs, have the ability to generalize across diverse tasks and adapt to human instructions. This paper investigates the potential of LLMs to facilitate planning in the context of human-robot collaborative tasks, with a focus on their ability to reason from high-level, vague human inputs, and fine-tune plans based on real-time feedback. We propose a novel hybrid framework that combines LLMs with human feedback to create dynamic, context-aware task plans. Our work also highlights how a single, concise prompt can be used for a wide range of tasks and environments, overcoming the limitations of long, detailed structured prompts typically used in prior studies. By integrating user preferences into the planning loop, we ensure that the generated plans are not only effective but aligned with human intentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01001v1">Towards An Efficient LLM Training Paradigm for CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated tremendous potential as the next-generation ranking-based recommendation system. Many recent works have shown that LLMs can significantly outperform conventional click-through-rate (CTR) prediction approaches. Despite such promising results, the computational inefficiency inherent in the current training paradigm makes it particularly challenging to train LLMs for ranking-based recommendation tasks on large datasets. To train LLMs for CTR prediction, most existing studies adopt the prevalent ''sliding-window'' paradigm. Given a sequence of $m$ user interactions, a unique training prompt is constructed for each interaction by designating it as the prediction target along with its preceding $n$ interactions serving as context. In turn, the sliding-window paradigm results in an overall complexity of $O(mn^2)$ that scales linearly with the length of user interactions. Consequently, a direct adoption to train LLMs with such strategy can result in prohibitively high training costs as the length of interactions grows. To alleviate the computational inefficiency, we propose a novel training paradigm, namely Dynamic Target Isolation (DTI), that structurally parallelizes the training of $k$ (where $k >> 1$) target interactions. Furthermore, we identify two major bottlenecks - hidden-state leakage and positional bias overfitting - that limit DTI to only scale up to a small value of $k$ (e.g., 5) then propose a computationally light solution to effectively tackle each. Through extensive experiments on three widely adopted public CTR datasets, we empirically show that DTI reduces training time by an average of $\textbf{92%}$ (e.g., from $70.5$ hrs to $5.31$ hrs), without compromising CTR prediction performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00912v1">HiBench: Benchmarking LLMs Capability on Hierarchical Structure Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Structure reasoning is a fundamental capability of large language models (LLMs), enabling them to reason about structured commonsense and answer multi-hop questions. However, existing benchmarks for structure reasoning mainly focus on horizontal and coordinate structures (\emph{e.g.} graphs), overlooking the hierarchical relationships within them. Hierarchical structure reasoning is crucial for human cognition, particularly in memory organization and problem-solving. It also plays a key role in various real-world tasks, such as information extraction and decision-making. To address this gap, we propose HiBench, the first framework spanning from initial structure generation to final proficiency assessment, designed to benchmark the hierarchical reasoning capabilities of LLMs systematically. HiBench encompasses six representative scenarios, covering both fundamental and practical aspects, and consists of 30 tasks with varying hierarchical complexity, totaling 39,519 queries. To evaluate LLMs comprehensively, we develop five capability dimensions that depict different facets of hierarchical structure understanding. Through extensive evaluation of 20 LLMs from 10 model families, we reveal key insights into their capabilities and limitations: 1) existing LLMs show proficiency in basic hierarchical reasoning tasks; 2) they still struggle with more complex structures and implicit hierarchical representations, especially in structural modification and textual reasoning. Based on these findings, we create a small yet well-designed instruction dataset, which enhances LLMs' performance on HiBench by an average of 88.84\% (Llama-3.1-8B) and 31.38\% (Qwen2.5-7B) across all tasks. The HiBench dataset and toolkit are available here, https://github.com/jzzzzh/HiBench, to encourage evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01926v1">Unnatural Languages Are Not Bugs but Features for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been observed to process non-human-readable text sequences, such as jailbreak prompts, often viewed as a bug for aligned LLMs. In this work, we present a systematic investigation challenging this perception, demonstrating that unnatural languages - strings that appear incomprehensible to humans but maintain semantic meanings for LLMs - contain latent features usable by models. Notably, unnatural languages possess latent features that can be generalized across different models and tasks during inference. Furthermore, models fine-tuned on unnatural versions of instruction datasets perform on-par with those trained on natural language, achieving 49.71 win rates in Length-controlled AlpacaEval 2.0 in average across various base models. In addition, through comprehensive analysis, we demonstrate that LLMs process unnatural languages by filtering noise and inferring contextual meaning from filtered words.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00858v1">Applying the Gricean Maxims to a Human-LLM Interaction Cycle: Design Insights from a Participatory Approach</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 Accepted CHI'25 LBW
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly used to assist users in various tasks through natural language interactions, these interactions often fall short due to LLMs' limited ability to infer contextual nuances and user intentions, unlike humans. To address this challenge, we draw inspiration from the Gricean Maxims--human communication theory that suggests principles of effective communication--and aim to derive design insights for enhancing human-AI interactions (HAI). Through participatory design workshops with communication experts, designers, and end-users, we identified ways to apply these maxims across the stages of the HAI cycle. Our findings include reinterpreted maxims tailored to human-LLM contexts and nine actionable design considerations categorized by interaction stage. These insights provide a concrete framework for designing more cooperative and user-centered LLM-based systems, bridging theoretical foundations in communication with practical applications in HAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00845v1">Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00813v1">HLoRA: Efficient Federated Learning System for LLM Heterogeneous Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Federated learning systems have been identified as an efficient approach to scaling distributed model training with a large amount of participants or data owners while guaranteeing data privacy. To apply the current most popular pre-trained large language models to other domains with data privacy guarantee requirements, existing works propose fine-tuning the pre-trained large language models in federated learning environments across data owners using the parameter efficient fine-tuning approaches, LoRA. To address the resource and data heterogeneous issues for the participants, previous works adopted heterogeneous LoRA using different ranks for different clients and pending their rank, which brings bias for the parameter aggregation. To address this issue, we propose HLoRA, an efficient federated learning system utilizing a modified LoRA approach that incorporates rank heterogeneity to optimize communication and computational efficiency. Experimental results, conducted using the Microsoft Research Paraphrase Corpus (MRPC), Quora Question Pairs (QQP) and Recognizing Textual Entailment (RTE), within the Plato federated learning framework, demonstrate that our method not only reduces resource demands but also outperforms traditional LoRA applications in terms of convergence speed and final model accuracy. This study shows that our approach can significantly improve the practical deployment of federated LLM fine-tuning, particularly in environments with diverse client resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00795v1">Towards Reliable LLM-Driven Fuzz Testing: Vision and Road Ahead</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Fuzz testing is a crucial component of software security assessment, yet its effectiveness heavily relies on valid fuzz drivers and diverse seed inputs. Recent advancements in Large Language Models (LLMs) offer transformative potential for automating fuzz testing (LLM4Fuzz), particularly in generating drivers and seeds. However, current LLM4Fuzz solutions face critical reliability challenges, including low driver validity rates and seed quality trade-offs, hindering their practical adoption. This paper aims to examine the reliability bottlenecks of LLM-driven fuzzing and explores potential research directions to address these limitations. It begins with an overview of the current development of LLM4SE and emphasizes the necessity for developing reliable LLM4Fuzz solutions. Following this, the paper envisions a vision where reliable LLM4Fuzz transforms the landscape of software testing and security for industry, software development practitioners, and economic accessibility. It then outlines a road ahead for future research, identifying key challenges and offering specific suggestions for the researchers to consider. This work strives to spark innovation in the field, positioning reliable LLM4Fuzz as a fundamental component of modern software testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00771v1">Evaluating Personalized Tool-Augmented LLMs from the Perspectives of Personalization and Proactivity</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      Personalized tool utilization is essential for aligning large language models (LLMs) with user preference in interaction scenarios with various tools. However, most of the current benchmarks primarily focus on either personalization of text generation or direct tool-utilizing, without considering both. In this work, we introduce a novel benchmark ETAPP for evaluating personalized tool invocation, establishing a sandbox environment, and a comprehensive dataset of 800 testing cases covering diverse user profiles. To improve the accuracy of our evaluation, we propose a key-point-based LLM evaluation method, mitigating biases in the LLM-as-a-judge system by manually annotating key points for each test case and providing them to LLM as the reference. Additionally, we evaluate the excellent LLMs and provide an in-depth analysis. Furthermore, we investigate the impact of different tool-invoking strategies on LLMs' personalization performance and the effects of fine-tuning in our task. The effectiveness of our preference-setting and key-point-based evaluation method is also validated. Our findings offer insights into improving personalized LLM agents. Our Code is available at https://github.com/hypasd-art/ETAPP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00767v1">LLMs are everywhere: Ubiquitous Utilization of AI Models through Air Computing</a></div>
    <div class="paper-meta">
      📅 2025-03-02
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      We are witnessing a new era where problem-solving and cognitive tasks are being increasingly delegated to Large Language Models (LLMs) across diverse domains, ranging from code generation to holiday planning. This trend also creates a demand for the ubiquitous execution of LLM-powered applications in a wide variety of environments in which traditional terrestrial 2D networking infrastructures may prove insufficient. A promising solution in this context is to extend edge computing into a 3D setting to include aerial platforms organized in multiple layers, a paradigm we refer to as air computing, to augment local devices for running LLM and Generative AI (GenAI) applications. This approach alleviates the strain on existing infrastructure while enhancing service efficiency by offloading computational tasks to the corresponding air units such as UAVs. Furthermore, the coordinated deployment of various air units can significantly improve the Quality of Experience (QoE) by ensuring seamless, adaptive, and resilient task execution. In this study, we investigate the synergy between LLM-based applications and air computing, exploring their potential across various use cases. Additionally, we present a disaster response case study demonstrating how the collaborative utilization of LLMs and air computing can significantly improve outcomes in critical situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00724v1">Unmasking Digital Falsehoods: A Comparative Analysis of LLM-Based Misinformation Detection Strategies</a></div>
    <div class="paper-meta">
      📅 2025-03-02
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation on social media has raised significant societal concerns, necessitating robust detection mechanisms. Large Language Models such as GPT-4 and LLaMA2 have been envisioned as possible tools for detecting misinformation based on their advanced natural language understanding and reasoning capabilities. This paper conducts a comparison of LLM-based approaches to detecting misinformation between text-based, multimodal, and agentic approaches. We evaluate the effectiveness of fine-tuned models, zero-shot learning, and systematic fact-checking mechanisms in detecting misinformation across different topic domains like public health, politics, and finance. We also discuss scalability, generalizability, and explainability of the models and recognize key challenges such as hallucination, adversarial attacks on misinformation, and computational resources. Our findings point towards the importance of hybrid approaches that pair structured verification protocols with adaptive learning techniques to enhance detection accuracy and explainability. The paper closes by suggesting potential avenues of future work, including real-time tracking of misinformation, federated learning, and cross-platform detection models.
    </details>
</div>
