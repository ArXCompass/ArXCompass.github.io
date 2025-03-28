# llm - 2025_03

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21760v1">MemInsight: Autonomous Memory Augmentation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have evolved to intelligently process information, make decisions, and interact with users or tools. A key capability is the integration of long-term memory capabilities, enabling these agents to draw upon historical interactions and knowledge. However, the growing memory size and need for semantic structuring pose significant challenges. In this work, we propose an autonomous memory augmentation approach, MemInsight, to enhance semantic data representation and retrieval mechanisms. By leveraging autonomous augmentation to historical interactions, LLM agents are shown to deliver more accurate and contextualized responses. We empirically validate the efficacy of our proposed approach in three task scenarios; conversational recommendation, question answering and event summarization. On the LLM-REDIAL dataset, MemInsight boosts persuasiveness of recommendations by up to 14%. Moreover, it outperforms a RAG baseline by 34% in recall for LoCoMo retrieval. Our empirical results show the potential of MemInsight to enhance the contextual performance of LLM agents across multiple tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21735v1">GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Ensuring the reliability and effectiveness of software release decisions is critical, particularly in safety-critical domains like automotive systems. Precise analysis of release validation data, often presented in tabular form, plays a pivotal role in this process. However, traditional methods that rely on manual analysis of extensive test datasets and validation metrics are prone to delays and high costs. Large Language Models (LLMs) offer a promising alternative but face challenges in analytical reasoning, contextual understanding, handling out-of-scope queries, and processing structured test data consistently; limitations that hinder their direct application in safety-critical scenarios. This paper introduces GateLens, an LLM-based tool for analyzing tabular data in the automotive domain. GateLens translates natural language queries into Relational Algebra (RA) expressions and then generates optimized Python code. It outperforms the baseline system on benchmarking datasets, achieving higher F1 scores and handling complex and ambiguous queries with greater robustness. Ablation studies confirm the critical role of the RA module, with performance dropping sharply when omitted. Industrial evaluations reveal that GateLens reduces analysis time by over 80% while maintaining high accuracy and reliability. As demonstrated by presented results, GateLens achieved high performance without relying on few-shot examples, showcasing strong generalization across various query types from diverse company roles. Insights from deploying GateLens with a partner automotive company offer practical guidance for integrating AI into critical workflows such as release validation. Results show that by automating test result analysis, GateLens enables faster, more informed, and dependable release decisions, and can thus advance software scalability and reliability in automotive systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18869v2">Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 9 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The efficiency of Large Language Model~(LLM) inference is often constrained by substantial memory bandwidth and capacity demands. Existing techniques, such as pruning, quantization, and mixture of experts/depth, reduce memory capacity and/or bandwidth consumption at the cost of slight degradation in inference quality. This paper introduces a design solution that further alleviates memory bottlenecks by enhancing the on-chip memory controller in AI accelerators to achieve two main objectives: (1) significantly reducing memory capacity and bandwidth usage through lossless block compression~(e.g., LZ4 and ZSTD) of model weights and key-value (KV) cache without compromising inference quality, and (2) enabling memory bandwidth and energy consumption to scale proportionally with context-dependent dynamic quantization. These goals are accomplished by equipping the on-chip memory controller with mechanisms to improve fine-grained bit-level accessibility and compressibility of weights and KV cache through LLM-aware configuration of in-memory placement and representation. Experimental results on publicly available LLMs demonstrate the effectiveness of this approach, showing memory footprint reductions of 25.2\% for model weights and 46.9\% for KV cache. In addition, our hardware prototype at 4\,GHz and 32 lanes (7\,nm) achieves 8\,TB/s throughput with a modest area overhead (under 3.8\,mm\(^2\)), which underscores the viability of LLM-aware memory control as a key to efficient large-scale inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05510v2">OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Temporal Awareness, the ability to reason dynamically based on the timestamp when a question is raised, is the key distinction between offline and online video LLMs. Unlike offline models, which rely on complete videos for static, post hoc analysis, online models process video streams incrementally and dynamically adapt their responses based on the timestamp at which the question is posed. Despite its significance, temporal awareness has not been adequately evaluated in existing benchmarks. To fill this gap, we present OVO-Bench (Online-VideO-Benchmark), a novel video benchmark that emphasizes the importance of timestamps for advanced online video understanding capability benchmarking. OVO-Bench evaluates the ability of video LLMs to reason and respond to events occurring at specific timestamps under three distinct scenarios: (1) Backward tracing: trace back to past events to answer the question. (2) Real-time understanding: understand and respond to events as they unfold at the current timestamp. (3) Forward active responding: delay the response until sufficient future information becomes available to answer the question accurately. OVO-Bench comprises 12 tasks, featuring 644 unique videos and approximately human-curated 2,800 fine-grained meta-annotations with precise timestamps. We combine automated generation pipelines with human curation. With these high-quality samples, we further developed an evaluation pipeline to systematically query video LLMs along the video timeline. Evaluations of nine Video-LLMs reveal that, despite advancements on traditional benchmarks, current models struggle with online video understanding, showing a significant gap compared to human agents. We hope OVO-Bench will drive progress in video LLMs and inspire future research in online video reasoning. Our benchmark and code can be accessed at https://github.com/JoeLeelyf/OVO-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21720v1">Collab: Controlled Decoding using Mixture of Agents for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Alignment of Large Language models (LLMs) is crucial for safe and trustworthy deployment in applications. Reinforcement learning from human feedback (RLHF) has emerged as an effective technique to align LLMs to human preferences and broader utilities, but it requires updating billions of model parameters, which is computationally expensive. Controlled Decoding, by contrast, provides a mechanism for aligning a model at inference time without retraining. However, single-agent decoding approaches often struggle to adapt to diverse tasks due to the complexity and variability inherent in these tasks. To strengthen the test-time performance w.r.t the target task, we propose a mixture of agent-based decoding strategies leveraging the existing off-the-shelf aligned LLM policies. Treating each prior policy as an agent in the spirit of mixture of agent collaboration, we develop a decoding method that allows for inference-time alignment through a token-level selection strategy among multiple agents. For each token, the most suitable LLM is dynamically chosen from a pool of models based on a long-term utility metric. This policy-switching mechanism ensures optimal model selection at each step, enabling efficient collaboration and alignment among LLMs during decoding. Theoretical analysis of our proposed algorithm establishes optimal performance with respect to the target task represented via a target reward for the given off-the-shelf models. We conduct comprehensive empirical evaluations with open-source aligned models on diverse tasks and preferences, which demonstrates the merits of this approach over single-agent decoding baselines. Notably, Collab surpasses the current SoTA decoding strategy, achieving an improvement of up to 1.56x in average reward and 71.89% in GPT-4 based win-tie rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21717v1">CLAIMCHECK: How Grounded are LLM Critiques of Scientific Papers?</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      A core part of scientific peer review involves providing expert critiques that directly assess the scientific claims a paper makes. While it is now possible to automatically generate plausible (if generic) reviews, ensuring that these reviews are sound and grounded in the papers' claims remains challenging. To facilitate LLM benchmarking on these challenges, we introduce CLAIMCHECK, an annotated dataset of NeurIPS 2023 and 2024 submissions and reviews mined from OpenReview. CLAIMCHECK is richly annotated by ML experts for weakness statements in the reviews and the paper claims that they dispute, as well as fine-grained labels of the validity, objectivity, and type of the identified weaknesses. We benchmark several LLMs on three claim-centric tasks supported by CLAIMCHECK, requiring models to (1) associate weaknesses with the claims they dispute, (2) predict fine-grained labels for weaknesses and rewrite the weaknesses to enhance their specificity, and (3) verify a paper's claims with grounded reasoning. Our experiments reveal that cutting-edge LLMs, while capable of predicting weakness labels in (2), continue to underperform relative to human experts on all other tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08180v2">Enhancing LLM Character-Level Manipulation via Divide and Conquer</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong generalization capabilities across a wide range of natural language processing (NLP) tasks. However, they exhibit notable weaknesses in character-level string manipulation, struggling with fundamental operations such as character deletion, insertion, and substitution. These challenges stem primarily from tokenization constraints, despite the critical role of such operations in data preprocessing and code generation. Through systematic analysis, we derive two key insights: (1) LLMs face significant difficulties in leveraging intrinsic token knowledge for character-level reasoning, and (2) atomized word structures can substantially enhance LLMs' ability to process token-level structural information. Building on these insights, we propose Character-Level Manipulation via Divide and Conquer, a novel approach designed to bridge the gap between token-level processing and character-level manipulation. Our method decomposes complex operations into explicit character-level subtasks coupled with controlled token reconstruction phases, leading to significant improvements in accuracy. Without additional training, our method significantly improves accuracies on the $\texttt{Deletion}$, $\texttt{Insertion}$, and $\texttt{Substitution}$ tasks. To support further research, we open-source our implementation and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01672v4">Practicing Stress Relief for the Everyday: Designing Social Simulation Using VR, AR, and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Stress is an inevitable part of day-to-day life yet many find themselves unable to manage it themselves, particularly when professional or peer support are not always readily available. As self-care becomes increasingly vital for mental well-being, this paper explores the potential of social simulation as a safe, virtual environment for practicing stress relief for everyday situations. Leveraging the immersive capabilities of VR, AR, and LLMs, we developed eight interactive prototypes for various everyday stressful scenarios (e.g. public speaking) then conducted prototype-driven semi-structured interviews with 19 participants. We reveal that people currently lack effective means to support themselves through everyday stress and found that social simulation fills a gap for simulating real environments for training mental health practices. We outline key considerations for future development of simulation for self-care, including risks of trauma from hyper-realism, distrust of LLM-recommended timing for mental health recommendations, and the value of accessibility for self-care interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21564v1">Cooking Task Planning using LLM and Verified by Graph Network</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Cooking tasks remain a challenging problem for robotics due to their complexity. Videos of people cooking are a valuable source of information for such task, but introduces a lot of variability in terms of how to translate this data to a robotic environment. This research aims to streamline this process, focusing on the task plan generation step, by using a Large Language Model (LLM)-based Task and Motion Planning (TAMP) framework to autonomously generate cooking task plans from videos with subtitles, and execute them. Conventional LLM-based task planning methods are not well-suited for interpreting the cooking video data due to uncertainty in the videos, and the risk of hallucination in its output. To address both of these problems, we explore using LLMs in combination with Functional Object-Oriented Networks (FOON), to validate the plan and provide feedback in case of failure. This combination can generate task sequences with manipulation motions that are logically correct and executable by a robot. We compare the execution of the generated plans for 5 cooking recipes from our approach against the plans generated by a few-shot LLM-only approach for a dual-arm robot setup. It could successfully execute 4 of the plans generated by our approach, whereas only 1 of the plans generated by solely using the LLM could be executed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17922v2">WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      With the advancements in long-context inference capabilities of large language models (LLMs), the KV cache has become one of the foundational components. However, its substantial GPU memory consumption makes KV cache compression a key technique for enabling efficient LLM inference in industrial scenarios. While recent studies have focused on optimizing the memory occupied by the KV cache, they overlook two critical factors: preserving semantic coherence and considering task-specific characteristic during compression. To address these limitations, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to task-specific characteristics, ensuring the retained KV cache captures continuous, essential context. Additionally, we introduce an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency. We rigorously evaluate WindowKV on the LongBench benchmark, and the results demonstrate that it maintains a performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Furthermore, our method also achieves state-of-the-art results in the Needle-in-a-Haystack evaluation, highlighting its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21480v1">OmniVox: Zero-Shot Emotion Recognition with Omni-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Submitted to COLM 2025. Preprint
    </div>
    <details class="paper-abstract">
      The use of omni-LLMs (large language models that accept any modality as input), particularly for multimodal cognitive state tasks involving speech, is understudied. We present OmniVox, the first systematic evaluation of four omni-LLMs on the zero-shot emotion recognition task. We evaluate on two widely used multimodal emotion benchmarks: IEMOCAP and MELD, and find zero-shot omni-LLMs outperform or are competitive with fine-tuned audio models. Alongside our audio-only evaluation, we also evaluate omni-LLMs on text only and text and audio. We present acoustic prompting, an audio-specific prompting strategy for omni-LLMs which focuses on acoustic feature analysis, conversation context analysis, and step-by-step reasoning. We compare our acoustic prompting to minimal prompting and full chain-of-thought prompting techniques. We perform a context window analysis on IEMOCAP and MELD, and find that using context helps, especially on IEMOCAP. We conclude with an error analysis on the generated acoustic reasoning outputs from the omni-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21422v1">From Deep Learning to LLMs: A survey of AI in Quantitative Investment</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Quantitative investment (quant) is an emerging, technology-driven approach in asset management, increasingy shaped by advancements in artificial intelligence. Recent advances in deep learning and large language models (LLMs) for quant finance have improved predictive modeling and enabled agent-based automation, suggesting a potential paradigm shift in this field. In this survey, taking alpha strategy as a representative example, we explore how AI contributes to the quantitative investment pipeline. We first examine the early stage of quant research, centered on human-crafted features and traditional statistical models with an established alpha pipeline. We then discuss the rise of deep learning, which enabled scalable modeling across the entire pipeline from data processing to order execution. Building on this, we highlight the emerging role of LLMs in extending AI beyond prediction, empowering autonomous agents to process unstructured data, generate alphas, and support self-iterative workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21393v1">An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study about the assessment of the quality of translations generated by LLMs, including Gemini, GPT and Google Translate. In this study, we address this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts that have been well translated by experts and use LLMs to generate their translations to English, and then we provide a comparison with selected expert (human) translations. Our findings suggest that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in figurative and philosophical contexts. The sentiment analysis revealed that GPT-4o and GPT-3.5 are better at preserving the sentiments for the Bhagavad Gita (Sanskrit-English) translations when compared to Google Translate. We observed a similar trend for the case of Tamas (Hindi-English) and Maha P (Telugu-English) translations. GPT-4o performs similarly to GPT-3.5 in the translation in terms of sentiments for the three languages. We found that LLMs are generally better at translation for capturing sentiments when compared to Google Translate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01877v2">Starjob: Dataset for LLM-Driven Job Shop Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 arXiv admin note: substantial text overlap with arXiv:2408.06993
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across various domains, but their potential for solving combinatorial optimization problems remains largely unexplored. In this paper, we investigate the applicability of LLMs to the Job Shop Scheduling Problem (JSSP), a classic challenge in combinatorial optimization that requires efficient job allocation to machines to minimize makespan. To this end, we introduce Starjob, the first supervised dataset for JSSP, comprising 130k instances specifically designed for training LLMs. Leveraging this dataset, we fine-tune the LLaMA 8B 4-bit quantized model with the LoRA method to develop an end-to-end scheduling approach. Our evaluation on standard benchmarks demonstrates that the proposed LLM-based method not only surpasses traditional Priority Dispatching Rules (PDRs) but also achieves notable improvements over state-of-the-art neural approaches like L2D, with an average improvement of 15.36% on DMU and 7.85% on Taillard benchmarks. These results highlight the untapped potential of LLMs in tackling combinatorial optimization problems, paving the way for future advancements in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21349v1">Fine-Tuning LLMs on Small Medical Datasets: Text Classification and Normalization Effectiveness on Cardiology reports and Discharge records</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 4 pages, 2 tables,
    </div>
    <details class="paper-abstract">
      We investigate the effectiveness of fine-tuning large language models (LLMs) on small medical datasets for text classification and named entity recognition tasks. Using a German cardiology report dataset and the i2b2 Smoking Challenge dataset, we demonstrate that fine-tuning small LLMs locally on limited training data can improve performance achieving comparable results to larger models. Our experiments show that fine-tuning improves performance on both tasks, with notable gains observed with as few as 200-300 training examples. Overall, the study highlights the potential of task-specific fine-tuning of LLMs for automating clinical workflows and efficiently extracting structured data from unstructured medical text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00493v2">Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Accepted by CVPR 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of Multimodal Large Language Models (MLLMs) has significantly impacted various multimodal tasks. However, these models face challenges in tasks that require spatial understanding within 3D environments. Efforts to enhance MLLMs, such as incorporating point cloud features, have been made, yet a considerable gap remains between the models' learned representations and the inherent complexity of 3D scenes. This discrepancy largely stems from the training of MLLMs on predominantly 2D data, which restricts their effectiveness in comprehending 3D spaces. To address this issue, in this paper, we propose a novel generalist model, i.e., Video-3D LLM, for 3D scene understanding. By treating 3D scenes as dynamic videos and incorporating 3D position encoding into these representations, our Video-3D LLM aligns video representations with real-world spatial contexts more accurately. In addition, we have implemented a maximum coverage sampling technique to optimize the trade-off between computational cost and performance. Extensive experiments demonstrate that our model achieves state-of-the-art performance on several 3D scene understanding benchmarks, including ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21248v1">ResearchBench: Benchmarking LLMs in Scientific Discovery via Inspiration-Based Task Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in assisting scientific research, yet their ability to discover high-quality research hypotheses remains unexamined due to the lack of a dedicated benchmark. To address this gap, we introduce the first large-scale benchmark for evaluating LLMs with a near-sufficient set of sub-tasks of scientific discovery: inspiration retrieval, hypothesis composition, and hypothesis ranking. We develop an automated framework that extracts critical components - research questions, background surveys, inspirations, and hypotheses - from scientific papers across 12 disciplines, with expert validation confirming its accuracy. To prevent data contamination, we focus exclusively on papers published in 2024, ensuring minimal overlap with LLM pretraining data. Our evaluation reveals that LLMs perform well in retrieving inspirations, an out-of-distribution task, suggesting their ability to surface novel knowledge associations. This positions LLMs as "research hypothesis mines", capable of facilitating automated scientific discovery by generating innovative hypotheses at scale with minimal human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21223v1">Rethinking Graph Structure Learning in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Recently, the emergence of large language models (LLMs) has prompted researchers to explore the integration of language descriptions into graphs, aiming to enhance model encoding capabilities from a data-centric perspective. This graph representation is called text-attributed graphs (TAGs). A review of prior advancements highlights that graph structure learning (GSL) is a pivotal technique for improving data utility, making it highly relevant to efficient TAG learning. However, most GSL methods are tailored for traditional graphs without textual information, underscoring the necessity of developing a new GSL paradigm. Despite clear motivations, it remains challenging: (1) How can we define a reasonable optimization objective for GSL in the era of LLMs, considering the massive parameters in LLM? (2) How can we design an efficient model architecture that enables seamless integration of LLM for this optimization objective? For Question 1, we reformulate existing GSL optimization objectives as a tree optimization framework, shifting the focus from obtaining a well-trained edge predictor to a language-aware tree sampler. For Question 2, we propose decoupled and training-free model design principles for LLM integration, shifting the focus from computation-intensive fine-tuning to more efficient inference. Based on this, we propose Large Language and Tree Assistant (LLaTA), which leverages tree-based LLM in-context learning to enhance the understanding of topology and text, enabling reliable inference and generating improved graph structure. Extensive experiments on 10 TAG datasets demonstrate that LLaTA enjoys flexibility - incorporated with any backbone; scalability - outperforms other LLM-based GSL methods in terms of running efficiency; effectiveness - achieves SOTA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18305v2">Enhancing LLM-based Code Translation in Repository Context via Triple Knowledge-Augmented</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in function-level code translation without repository-level context. However, the performance of LLMs in repository-level context code translation remains suboptimal due to complex dependencies and context, hindering their adoption in industrial settings. In this work, we propose a novel LLM-based code translation technique K-Trans, which leverages triple knowledge augmentation to enhance LLM's translation quality under repository context in real-world software development. First, K-Trans constructs a translation knowledge base by extracting relevant information from target-language codebases, the repository being translated, and prior translation results. Second, for each function to be translated, K-Trans retrieves relevant triple knowledge, including target-language code samples, dependency usage examples, and successful translation function pairs, serving as references to enhance LLM for translation. Third, K-Trans constructs a knowledge-augmented translation prompt using the retrieved triple knowledge and employs LLMs to generate the translated code while preserving repository context. It further leverages LLMs for self-debugging, enhancing translation correctness. The experiments show that K-Trans substantially outperforms the baseline adapted from previous work by 19.4%/40.2% relative improvement in pass@1 and 0.138 in CodeBLEU. It is important to note that the results also demonstrate that each knowledge significantly contributes to K-Trans's effectiveness in handling repository-level context code translation, with dependency usage examples making the most notable contribution. Moreover, as the self-evolution process progresses, the knowledge base continuously enhances the LLM's performance across various aspects of the repository-level code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14963v5">Achieving >97% on GSM8K: Deeply Understanding the Problems Makes LLMs Better Solvers for Math Word Problems</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: { 10.1007/s11704-025-41102-z }
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has enhanced the performance of Large Language Models (LLMs) across various reasoning tasks. However, CoT still falls short in dealing with complex math word problems, as it usually suffers from three pitfalls: semantic misunderstanding errors, calculation errors, and step-missing errors. Prior studies involve addressing the calculation errors and step-missing errors, but neglect the semantic misunderstanding errors, which is the major factor limiting the reasoning performance of LLMs. To this end, we propose a simple-yet-effective method, namely Deeply Understanding the Problems (DUP), to improve the LLMs' math problem-solving ability by addressing semantic misunderstanding errors. The core of our method is to encourage the LLMs to deeply understand the problems and extract the key problem-solving information used for better reasoning. Extensive experiments on 10 diverse reasoning benchmarks show that our DUP method consistently outperforms the other counterparts by a large margin. More encouragingly, DUP achieves a new SOTA result on the GSM8K benchmark, with an accuracy of 97.1% under the zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09056v3">Adapting Language-Specific LLMs to a Reasoning Model in One Day via Model Merging -- An Open Recipe</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      This paper investigates data selection and model merging methodologies aimed at incorporating advanced reasoning capabilities such as those of DeepSeek R1 into language-specific large language models (LLMs), with a particular focus on the Thai LLM. Our goal is to enhance the reasoning capabilities of language-specific LLMs while maintaining their target language abilities. DeepSeek R1 excels in reasoning but primarily benefits high-resource languages such as English and Chinese. However, low-resource languages remain underserved due to the dominance of English-centric training data and model optimizations, which limit performance in these languages. This limitation results in unreliable code-switching and diminished effectiveness on tasks in low-resource languages. Meanwhile, local and regional LLM initiatives have attempted to bridge this gap by developing language-specific LLMs that focus on improving local linguistic fidelity. We demonstrate that, with only publicly available datasets and a computational budget of $120, it is possible to enhance the reasoning capabilities of language-specific LLMs to match the level of DeepSeek R1, without compromising their performance on target language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21190v1">Leveraging LLMs with Iterative Loop Structure for Enhanced Social Intelligence in Video Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Social intelligence, the ability to interpret emotions, intentions, and behaviors, is essential for effective communication and adaptive responses. As robots and AI systems become more prevalent in caregiving, healthcare, and education, the demand for AI that can interact naturally with humans grows. However, creating AI that seamlessly integrates multiple modalities, such as vision and speech, remains a challenge. Current video-based methods for social intelligence rely on general video recognition or emotion recognition techniques, often overlook the unique elements inherent in human interactions. To address this, we propose the Looped Video Debating (LVD) framework, which integrates Large Language Models (LLMs) with visual information, such as facial expressions and body movements, to enhance the transparency and reliability of question-answering tasks involving human interaction videos. Our results on the Social-IQ 2.0 benchmark show that LVD achieves state-of-the-art performance without fine-tuning. Furthermore, supplementary human annotations on existing datasets provide insights into the model's accuracy, guiding future improvements in AI-driven social intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19470v2">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21155v1">Embedding Domain-Specific Knowledge from LLMs into the Feature Engineering Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 9 pages, 4 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Feature engineering is mandatory in the machine learning pipeline to obtain robust models. While evolutionary computation is well-known for its great results both in feature selection and feature construction, its methods are computationally expensive due to the large number of evaluations required to induce the final model. Part of the reason why these algorithms require a large number of evaluations is their lack of domain-specific knowledge, resulting in a lot of random guessing during evolution. In this work, we propose using Large Language Models (LLMs) as an initial feature construction step to add knowledge to the dataset. By doing so, our results show that the evolution can converge faster, saving us computational resources. The proposed approach only provides the names of the features in the dataset and the target objective to the LLM, making it usable even when working with datasets containing private data. While consistent improvements to test performance were only observed for one-third of the datasets (CSS, PM, and IM10), possibly due to problems being easily explored by LLMs, this approach only decreased the model performance in 1/77 test cases. Additionally, this work introduces the M6GP feature engineering algorithm to symbolic regression, showing it can improve the results of the random forest regressor and produce competitive results with its predecessor, M3GP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00590v2">Characterizing LLM-Empowered Personalized Story-Reading and Interaction for Children: Insights from Multi-Stakeholder Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 Accepted at CHI 2025
    </div>
    <details class="paper-abstract">
      Personalized interaction is highly valued by parents in their story-reading activities with children. While AI-empowered story-reading tools have been increasingly used, their abilities to support personalized interaction with children are still limited. Recent advances in large language models (LLMs) show promise in facilitating personalized interactions, but little is known about how to effectively and appropriately use LLMs to enhance children's personalized story-reading experiences. This work explores this question through a design-based study. Drawing on a formative study, we designed and developed StoryMate, an LLM-empowered personalized interactive story-reading tool for children, following an empirical study with children, parents, and education experts. Our participants valued the personalized features in StoryMate, and also highlighted the need to support personalized content, guiding mechanisms, reading context variations, and interactive interfaces. Based on these findings, we propose a series of design recommendations for better using LLMs to empower children's personalized story reading and interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00393v4">Unleashing the Power of LLM to Infer State Machine from the Protocol Implementation</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      State machines are essential for enhancing protocol analysis to identify vulnerabilities. However, inferring state machines from network protocol implementations is challenging due to complex code syntax and semantics. Traditional dynamic analysis methods often miss critical state transitions due to limited coverage, while static analysis faces path explosion issues. To overcome these challenges, we introduce a novel state machine inference approach utilizing Large Language Models (LLMs), named ProtocolGPT. This method employs retrieval augmented generation technology to enhance a pre-trained model with specific knowledge from protocol implementations. Through effective prompt engineering, we accurately identify and infer state machines. To the best of our knowledge, our approach represents the first state machine inference that leverages the source code of protocol implementations. Our evaluation of six protocol implementations shows that our method achieves a precision of over 90%, outperforming the baselines by more than 30%. Furthermore, integrating our approach with protocol fuzzing improves coverage by more than 20% and uncovers two 0-day vulnerabilities compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v2">Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21098v1">Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Generative retrieval (GR) has revolutionized document retrieval with the advent of large language models (LLMs), and LLM-based GR is gradually being adopted by the industry. Despite its remarkable advantages and potential, LLM-based GR suffers from hallucination and generates documents that are irrelevant to the query in some instances, severely challenging its credibility in practical applications. We thereby propose an optimized GR framework designed to alleviate retrieval hallucination, which integrates knowledge distillation reasoning in model training and incorporate decision agent to further improve retrieval precision. Specifically, we employ LLMs to assess and reason GR retrieved query-document (q-d) pairs, and then distill the reasoning data as transferred knowledge to the GR model. Moreover, we utilize a decision agent as post-processing to extend the GR retrieved documents through retrieval model and select the most relevant ones from multi perspectives as the final generative retrieval result. Extensive offline experiments on real-world datasets and online A/B tests on Fund Search and Insurance Search in Alipay demonstrate our framework's superiority and effectiveness in improving search quality and conversion gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13768v2">Evaluation-Driven Development of LLM Agents: A Process Model and Reference Architecture</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled the emergence of LLM agents: autonomous systems capable of achieving under-specified goals and adapting post-deployment, often without explicit code or model changes. Evaluating these agents is critical to ensuring their performance and safety, especially given their dynamic, probabilistic, and evolving nature. However, traditional approaches such as predefined test cases and standard redevelopment pipelines struggle to address the unique challenges of LLM agent evaluation. These challenges include capturing open-ended behaviors, handling emergent outcomes, and enabling continuous adaptation over the agent's lifecycle. To address these issues, we propose an evaluation-driven development approach, inspired by test-driven and behavior-driven development but reimagined for the unique characteristics of LLM agents. Through a multivocal literature review (MLR), we synthesize the limitations of existing LLM evaluation methods and introduce a novel process model and reference architecture tailored for evaluation-driven development of LLM agents. Our approach integrates online (runtime) and offline (redevelopment) evaluations, enabling adaptive runtime adjustments and systematic iterative refinement of pipelines, artifacts, system architecture, and LLMs themselves. By continuously incorporating evaluation results, including fine-grained feedback from human and AI evaluators, into each stage of development and operation, this framework ensures that LLM agents remain aligned with evolving goals, user needs, and governance standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14340v2">MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-27
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Maintaining and scaling software systems relies heavily on effective code refactoring, yet this process remains labor-intensive, requiring developers to carefully analyze existing codebases and prevent the introduction of new defects. Although recent advancements have leveraged Large Language Models (LLMs) to automate refactoring tasks, current solutions are constrained in scope and lack mechanisms to guarantee code compilability and successful test execution. In this work, we introduce MANTRA, a comprehensive LLM agent-based framework that automates method-level refactoring. MANTRA integrates Context-Aware Retrieval-Augmented Generation, coordinated Multi-Agent Collaboration, and Verbal Reinforcement Learning to emulate human decision-making during refactoring while preserving code correctness and readability. Our empirical study, conducted on 703 instances of "pure refactorings" (i.e., code changes exclusively involving structural improvements), drawn from 10 representative Java projects, covers the six most prevalent refactoring operations. Experimental results demonstrate that MANTRA substantially surpasses a baseline LLM model (RawGPT ), achieving an 82.8% success rate (582/703) in producing code that compiles and passes all tests, compared to just 8.7% (61/703) with RawGPT. Moreover, in comparison to IntelliJ's LLM-powered refactoring tool (EM-Assist), MANTRA exhibits a 50% improvement in generating Extract Method transformations. A usability study involving 37 professional developers further shows that refactorings performed by MANTRA are perceived to be as readable and reusable as human-written code, and in certain cases, even more favorable. These results highlight the practical advantages of MANTRA and emphasize the growing potential of LLM-based systems in advancing the automation of software refactoring tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21080v1">EQ-Negotiator: An Emotion-Reasoning LLM Agent in Credit Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-03-27
    </div>
    <details class="paper-abstract">
      While large language model (LLM)-based chatbots have been applied for effective engagement in credit dialogues, their capacity for dynamic emotional expression remains limited. Current agents primarily rely on passive empathy rather than affective reasoning. For instance, when faced with persistent client negativity, the agent should employ strategic emotional adaptation by expressing measured anger to discourage counterproductive behavior and guide the conversation toward resolution. This context-aware emotional modulation is essential for imitating the nuanced decision-making of human negotiators. This paper introduces an EQ-negotiator that combines emotion sensing from pre-trained language models (PLMs) with emotional reasoning based on Game Theory and Hidden Markov Models. It takes into account both the current and historical emotions of the client to better manage and address negative emotions during interactions. By fine-tuning pre-trained language models (PLMs) on public emotion datasets and validating them on the credit dialogue datasets, our approach enables LLM-based agents to effectively capture shifts in client emotions and dynamically adjust their response tone based on our emotion decision policies in real-world financial negotiations. This EQ-negotiator can also help credit agencies foster positive client relationships, enhancing satisfaction in credit services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v1">Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20666v1">TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs for Clinical Interviews</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Submitted to the American Medical Informatics Association (AMIA) 2025 Annual Symposium, 10 pages
    </div>
    <details class="paper-abstract">
      Thematic analysis (TA) is a widely used qualitative approach for uncovering latent meanings in unstructured text data. TA provides valuable insights in healthcare but is resource-intensive. Large Language Models (LLMs) have been introduced to perform TA, yet their applications in healthcare remain unexplored. Here, we propose TAMA: A Human-AI Collaborative Thematic Analysis framework using Multi-Agent LLMs for clinical interviews. We leverage the scalability and coherence of multi-agent systems through structured conversations between agents and coordinate the expertise of cardiac experts in TA. Using interview transcripts from parents of children with Anomalous Aortic Origin of a Coronary Artery (AAOCA), a rare congenital heart disease, we demonstrate that TAMA outperforms existing LLM-assisted TA approaches, achieving higher thematic hit rate, coverage, and distinctiveness. TAMA demonstrates strong potential for automated TA in clinical settings by leveraging multi-agent LLM systems with human-in-the-loop integration by enhancing quality while significantly reducing manual workload.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20641v1">Unlocking Efficient Long-to-Short LLM Reasoning with Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Work in progress; technical report
    </div>
    <details class="paper-abstract">
      The transition from System 1 to System 2 reasoning in large language models (LLMs) has marked significant advancements in handling complex tasks through deliberate, iterative thinking. However, this progress often comes at the cost of efficiency, as models tend to overthink, generating redundant reasoning steps without proportional improvements in output quality. Long-to-Short (L2S) reasoning has emerged as a promising solution to this challenge, aiming to balance reasoning depth with practical efficiency. While existing approaches, such as supervised fine-tuning (SFT), reinforcement learning (RL), and prompt engineering, have shown potential, they are either computationally expensive or unstable. Model merging, on the other hand, offers a cost-effective and robust alternative by integrating the quick-thinking capabilities of System 1 models with the methodical reasoning of System 2 models. In this work, we present a comprehensive empirical study on model merging for L2S reasoning, exploring diverse methodologies, including task-vector-based, SVD-based, and activation-informed merging. Our experiments reveal that model merging can reduce average response length by up to 55% while preserving or even improving baseline performance. We also identify a strong correlation between model scale and merging efficacy with extensive evaluations on 1.5B/7B/14B/32B models. Furthermore, we investigate the merged model's ability to self-critique and self-correct, as well as its adaptive response length based on task complexity. Our findings highlight model merging as a highly efficient and effective paradigm for L2S reasoning, offering a practical solution to the overthinking problem while maintaining the robustness of System 2 reasoning. This work can be found on Github https://github.com/hahahawu/Long-to-Short-via-Model-Merging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20623v1">Collaborative Storytelling and LLM: A Linguistic Analysis of Automatically-Generated Role-Playing Game Sessions</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Role-playing games (RPG) are games in which players interact with one another to create narratives. The role of players in the RPG is largely based on the interaction between players and their characters. This emerging form of shared narrative, primarily oral, is receiving increasing attention. In particular, many authors investigated the use of an LLM as an actor in the game. In this paper, we aim to discover to what extent the language of Large Language Models (LLMs) exhibit oral or written features when asked to generate an RPG session without human interference. We will conduct a linguistic analysis of the lexical and syntactic features of the generated texts and compare the results with analyses of conversations, transcripts of human RPG sessions, and books. We found that LLMs exhibit a pattern that is distinct from all other text categories, including oral conversations, human RPG sessions and books. Our analysis has shown how training influences the way LLMs express themselves and provides important indications of the narrative capabilities of these tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20552v1">Injecting Adrenaline into LLM Serving: Boosting Resource Utilization and Throughput via Attention Disaggregation</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 14 pages, 18 figures
    </div>
    <details class="paper-abstract">
      In large language model (LLM) serving systems, executing each request consists of two phases: the compute-intensive prefill phase and the memory-intensive decoding phase. To prevent performance interference between the two phases, current LLM serving systems typically adopt prefill-decoding disaggregation, where the two phases are split across separate machines. However, we observe this approach leads to significant resource underutilization. Specifically, prefill instances that are compute-intensive suffer from low memory utilization, while decoding instances that are memory-intensive experience low compute utilization. To address this problem, this paper proposes Adrenaline, an attention disaggregation and offloading mechanism designed to enhance resource utilization and performance in LLM serving systems. Adrenaline's key innovation lies in disaggregating part of the attention computation in the decoding phase and offloading them to prefill instances. The memory-bound nature of decoding-phase attention computation inherently enables an effective offloading strategy, yielding two complementary advantages: 1) improved memory capacity and bandwidth utilization in prefill instances, and 2) increased decoding batch sizes that enhance compute utilization in decoding instances, collectively boosting overall system performance. Adrenaline achieves these gains through three key techniques: low-latency decoding synchronization, resource-efficient prefill colocation, and load-aware offloading scheduling. Experimental results show that Adrenaline achieves 2.28x higher memory capacity and 2.07x better memory bandwidth utilization in prefill instances, up to 1.67x improvements in compute utilization for decoding instances, and 1.68x higher overall inference throughput compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10927v2">OASST-ETC Dataset: Alignment Signals from Eye-tracking Analysis of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 This paper has been accepted to ACM ETRA 2025 and published on PACMHCI
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have significantly advanced natural language processing, aligning them with human preferences remains an open challenge. Although current alignment methods rely primarily on explicit feedback, eye-tracking (ET) data offers insights into real-time cognitive processing during reading. In this paper, we present OASST-ETC, a novel eye-tracking corpus capturing reading patterns from 24 participants, while evaluating LLM-generated responses from the OASST1 dataset. Our analysis reveals distinct reading patterns between preferred and non-preferred responses, which we compare with synthetic eye-tracking data. Furthermore, we examine the correlation between human reading measures and attention patterns from various transformer-based models, discovering stronger correlations in preferred responses. This work introduces a unique resource for studying human cognitive processing in LLM evaluation and suggests promising directions for incorporating eye-tracking data into alignment methods. The dataset and analysis code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15133v2">Don't Use LLMs to Make Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Making the relevance judgments for a TREC-style test collection can be complex and expensive. A typical TREC track usually involves a team of six contractors working for 2-4 weeks. Those contractors need to be trained and monitored. Software has to be written to support recording relevance judgments correctly and efficiently. The recent advent of large language models that produce astoundingly human-like flowing text output in response to a natural language prompt has inspired IR researchers to wonder how those models might be used in the relevance judgment collection process. At the ACM SIGIR 2024 conference, a workshop ``LLM4Eval'' provided a venue for this work, and featured a data challenge activity where participants reproduced TREC deep learning track judgments, as was done by Thomas et al (arXiv:2408.08896, arXiv:2309.10621). I was asked to give a keynote at the workshop, and this paper presents that keynote in article form. The bottom-line-up-front message is, don't use LLMs to create relevance judgments for TREC-style evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20518v1">Exploring the Effect of Robotic Embodiment and Empathetic Tone of LLMs on Empathy Elicitation</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 *Liza Darwesh, Jaspreet Singh, Marin Marian, and Eduard Alexa contributed equally to this work.*
    </div>
    <details class="paper-abstract">
      This study investigates the elicitation of empathy toward a third party through interaction with social agents. Participants engaged with either a physical robot or a voice-enabled chatbot, both driven by a large language model (LLM) programmed to exhibit either an empathetic tone or remain neutral. The interaction is focused on a fictional character, Katie Banks, who is in a challenging situation and in need of financial donations. The willingness to help Katie, measured by the number of hours participants were willing to volunteer, along with their perceptions of the agent, were assessed for 60 participants. Results indicate that neither robotic embodiment nor empathetic tone significantly influenced participants' willingness to volunteer. While the LLM effectively simulated human empathy, fostering genuine empathetic responses in participants proved challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20263v1">L4: Diagnosing Large-scale LLM Training Failures via Automated Log Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 To appear in companion proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering (FSE'25). 13 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) show their capabilities across various applications, training customized LLMs has become essential for modern enterprises. However, due to the complexity of LLM training, which requires massive computational resources and extensive training time, failures are inevitable during the training process. These failures result in considerable waste of resource and time, highlighting the critical need for effective and efficient failure diagnosis to reduce the cost of LLM training. In this paper, we present the first empirical study on the failure reports of 428 LLM training failures in our production Platform-X between May 2023 and April 2024. Our study reveals that hardware and user faults are the predominant root causes, and current diagnosis processes rely heavily on training logs. Unfortunately, existing log-based diagnostic methods fall short in handling LLM training logs. Considering the unique features of LLM training, we identify three distinct patterns of LLM training logs: cross-job, spatial, and temporal patterns. We then introduce our Log-based Large-scale LLM training failure diagnosis framework, L4, which can automatically extract failure-indicating information (i.e., log events, nodes, stages, and iterations) from extensive training logs, thereby reducing manual effort and facilitating failure recovery. Experimental results on real-world datasets show that L4 outperforms existing approaches in identifying failure-indicating logs and localizing faulty nodes. Furthermore, L4 has been applied in Platform-X and demonstrated its effectiveness in enabling accurate and efficient failure diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20241v1">LGR: LLM-Guided Ranking of Frontiers for Object Goal Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 10 pages, 11 figures, technical report
    </div>
    <details class="paper-abstract">
      Object Goal Navigation (OGN) is a fundamental task for robots and AI, with key applications such as mobile robot image databases (MRID). In particular, mapless OGN is essential in scenarios involving unknown or dynamic environments. This study aims to enhance recent modular mapless OGN systems by leveraging the commonsense reasoning capabilities of large language models (LLMs). Specifically, we address the challenge of determining the visiting order in frontier-based exploration by framing it as a frontier ranking problem. Our approach is grounded in recent findings that, while LLMs cannot determine the absolute value of a frontier, they excel at evaluating the relative value between multiple frontiers viewed within a single image using the view image as context. We dynamically manage the frontier list by adding and removing elements, using an LLM as a ranking model. The ranking results are represented as reciprocal rank vectors, which are ideal for multi-view, multi-query information fusion. We validate the effectiveness of our method through evaluations in Habitat-Sim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20228v1">TeleLoRA: Teleporting Model-Specific Alignment Across LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20226v1">Raising Awareness of Location Information Vulnerabilities in Social Media Photos using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Published at ACM CHI 2025 Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Location privacy leaks can lead to unauthorised tracking, identity theft, and targeted attacks, compromising personal security and privacy. This study explores LLM-powered location privacy leaks associated with photo sharing on social media, focusing on user awareness, attitudes, and opinions. We developed and introduced an LLM-powered location privacy intervention app to 19 participants, who used it over a two-week period. The app prompted users to reflect on potential privacy leaks that a widely available LLM could easily detect, such as visual landmarks & cues that could reveal their location, and provided ways to conceal this information. Through in-depth interviews, we found that our intervention effectively increased users' awareness of location privacy and the risks posed by LLMs. It also encouraged users to consider the importance of maintaining control over their privacy data and sparked discussions about the future of location privacy-preserving technologies. Based on these insights, we offer design implications to support the development of future user-centred, location privacy-preserving technologies for social media photos.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v1">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20182v1">Leveraging Implicit Sentiments: Enhancing Reliability and Validity in Psychological Trait Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Code available via https://github.com/dependentsign/CSI
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to their increasing integration into human life. With the transition from mere tools to human-like assistants, understanding their psychological aspects-such as emotional tendencies and personalities-becomes essential for ensuring their trustworthiness. However, current psychological evaluations of LLMs, often based on human psychological assessments like the BFI, face significant limitations. The results from these approaches often lack reliability and have limited validity when predicting LLM behavior in real-world scenarios. In this work, we introduce a novel evaluation instrument specifically designed for LLMs, called Core Sentiment Inventory (CSI). CSI is a bilingual tool, covering both English and Chinese, that implicitly evaluates models' sentiment tendencies, providing an insightful psychological portrait of LLM across three dimensions: optimism, pessimism, and neutrality. Through extensive experiments, we demonstrate that: 1) CSI effectively captures nuanced emotional patterns, revealing significant variation in LLMs across languages and contexts; 2) Compared to current approaches, CSI significantly improves reliability, yielding more consistent results; and 3) The correlation between CSI scores and the sentiment of LLM's real-world outputs exceeds 0.85, demonstrating its strong validity in predicting LLM behavior. We make CSI public available via: https://github.com/dependentsign/CSI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v4">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency. Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v2">Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) handle increasingly longer contexts, serving inference requests for context lengths in the range of millions of tokens presents unique challenges. While existing techniques are effective for training, they fail to address the unique challenges of inference, such as varying prefill and decode phases and their associated latency constraints -- like Time to First Token (TTFT) and Time per Output Token (TPOT). Furthermore, no long-context inference solutions address head-of-line blocking today. We present Medha, a system for efficient long-context LLM inference that introduces three key innovations: adaptive chunking with slack-aware scheduling to prevent head-ofline blocking, Sequence Pipeline Parallelism (SPP) to reduce TTFT, and KV Cache Parallelism (KVP) to minimize TPOT. By combining these into a novel 3D parallelism serving engine, Medha achieves unprecedented scale -- supporting contexts up to 10M tokens with production-grade latency. Our evaluation shows Medha reduces median latency by up to 30x compared to state-of-the-art systems when serving a mix of short and long requests, while improving throughput by upwards of 5x. This enables, for the first time, efficient long-context LLM inference at scale without compromising on shorter request latencies or system efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20126v1">Can We Make Code Green? Understanding Trade-Offs in LLMs vs. Human Code Optimizations</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      The rapid technological evolution has accelerated software development for various domains and use cases, contributing to a growing share of global carbon emissions. While recent large language models (LLMs) claim to assist developers in optimizing code for performance and energy efficiency, their efficacy in real-world scenarios remains under exploration. In this work, we explore the effectiveness of LLMs in reducing the environmental footprint of real-world projects, focusing on software written in Matlab-widely used in both academia and industry for scientific and engineering applications. We analyze energy-focused optimization on 400 scripts across 100 top GitHub repositories. We examine potential 2,176 optimizations recommended by leading LLMs, such as GPT-3, GPT-4, Llama, and Mixtral, and a senior Matlab developer, on energy consumption, memory usage, execution time consumption, and code correctness. The developer serves as a real-world baseline for comparing typical human and LLM-generated optimizations. Mapping these optimizations to 13 high-level themes, we found that LLMs propose a broad spectrum of improvements--beyond energy efficiency--including improving code readability and maintainability, memory management, error handling while the developer overlooked some parallel processing, error handling etc. However, our statistical tests reveal that the energy-focused optimizations unexpectedly negatively impacted memory usage, with no clear benefits regarding execution time or energy consumption. Our qualitative analysis of energy-time trade-offs revealed that some themes, such as vectorization preallocation, were among the common themes shaping these trade-offs. With LLMs becoming ubiquitous in modern software development, our study serves as a call to action: prioritizing the evaluation of common coding practices to identify the green ones.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20981v1">Patients Speak, AI Listens: LLM-based Analysis of Online Reviews Uncovers Key Drivers for Urgent Care Satisfaction</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Investigating the public experience of urgent care facilities is essential for promoting community healthcare development. Traditional survey methods often fall short due to limited scope, time, and spatial coverage. Crowdsourcing through online reviews or social media offers a valuable approach to gaining such insights. With recent advancements in large language models (LLMs), extracting nuanced perceptions from reviews has become feasible. This study collects Google Maps reviews across the DMV and Florida areas and conducts prompt engineering with the GPT model to analyze the aspect-based sentiment of urgent care. We first analyze the geospatial patterns of various aspects, including interpersonal factors, operational efficiency, technical quality, finances, and facilities. Next, we determine Census Block Group(CBG)-level characteristics underpinning differences in public perception, including population density, median income, GINI Index, rent-to-income ratio, household below poverty rate, no insurance rate, and unemployment rate. Our results show that interpersonal factors and operational efficiency emerge as the strongest determinants of patient satisfaction in urgent care, while technical quality, finances, and facilities show no significant independent effects when adjusted for in multivariate models. Among socioeconomic and demographic factors, only population density demonstrates a significant but modest association with patient ratings, while the remaining factors exhibit no significant correlations. Overall, this study highlights the potential of crowdsourcing to uncover the key factors that matter to residents and provide valuable insights for stakeholders to improve public satisfaction with urgent care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20953v1">Clean & Clear: Feasibility of Safe LLM Clinical Guidance</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Background: Clinical guidelines are central to safe evidence-based medicine in modern healthcare, providing diagnostic criteria, treatment options and monitoring advice for a wide range of illnesses. LLM-empowered chatbots have shown great promise in Healthcare Q&A tasks, offering the potential to provide quick and accurate responses to medical inquiries. Our main objective was the development and preliminary assessment of an LLM-empowered chatbot software capable of reliably answering clinical guideline questions using University College London Hospital (UCLH) clinical guidelines. Methods: We used the open-weight Llama-3.1-8B LLM to extract relevant information from the UCLH guidelines to answer questions. Our approach highlights the safety and reliability of referencing information over its interpretation and response generation. Seven doctors from the ward assessed the chatbot's performance by comparing its answers to the gold standard. Results: Our chatbot demonstrates promising performance in terms of relevance, with ~73% of its responses rated as very relevant, showcasing a strong understanding of the clinical context. Importantly, our chatbot achieves a recall of 0.98 for extracted guideline lines, substantially minimising the risk of missing critical information. Approximately 78% of responses were rated satisfactory in terms of completeness. A small portion (~14.5%) contained minor unnecessary information, indicating occasional lapses in precision. The chatbot' showed high efficiency, with an average completion time of 10 seconds, compared to 30 seconds for human respondents. Evaluation of clinical reasoning showed that 72% of the chatbot's responses were without flaws. Our chatbot demonstrates significant potential to speed up and improve the process of accessing locally relevant clinical information for healthcare professionals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20934v1">Leveraging LLMs, IDEs, and Semantic Embeddings for Automated Move Method Refactoring</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      MOVEMETHOD is a hallmark refactoring. Despite a plethora of research tools that recommend which methods to move and where, these recommendations do not align with how expert developers perform MOVEMETHOD. Given the extensive training of Large Language Models and their reliance upon naturalness of code, they should expertly recommend which methods are misplaced in a given class and which classes are better hosts. Our formative study of 2016 LLM recommendations revealed that LLMs give expert suggestions, yet they are unreliable: up to 80% of the suggestions are hallucinations. We introduce the first LLM fully powered assistant for MOVEMETHOD refactoring that automates its whole end-to-end lifecycle, from recommendation to execution. We designed novel solutions that automatically filter LLM hallucinations using static analysis from IDEs and a novel workflow that requires LLMs to be self-consistent, critique, and rank refactoring suggestions. As MOVEMETHOD refactoring requires global, projectlevel reasoning, we solved the limited context size of LLMs by employing refactoring-aware retrieval augment generation (RAG). Our approach, MM-assist, synergistically combines the strengths of the LLM, IDE, static analysis, and semantic relevance. In our thorough, multi-methodology empirical evaluation, we compare MM-assist with the previous state-of-the-art approaches. MM-assist significantly outperforms them: (i) on a benchmark widely used by other researchers, our Recall@1 and Recall@3 show a 1.7x improvement; (ii) on a corpus of 210 recent refactorings from Open-source software, our Recall rates improve by at least 2.4x. Lastly, we conducted a user study with 30 experienced participants who used MM-assist to refactor their own code for one week. They rated 82.8% of MM-assist recommendations positively. This shows that MM-assist is both effective and useful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20851v1">StepGrade: Grading Programming Assignments with Context-Aware LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Accepted to the 15th IEEE Integrated STEM Education Conference (ISEC)
    </div>
    <details class="paper-abstract">
      Grading programming assignments is a labor-intensive and time-consuming process that demands careful evaluation across multiple dimensions of the code. To overcome these challenges, automated grading systems are leveraged to enhance efficiency and reduce the workload on educators. Traditional automated grading systems often focus solely on correctness, failing to provide interpretable evaluations or actionable feedback for students. This study introduces StepGrade, which explores the use of Chain-of-Thought (CoT) prompting with Large Language Models (LLMs) as an innovative solution to address these challenges. Unlike regular prompting, which offers limited and surface-level outputs, CoT prompting allows the model to reason step-by-step through the interconnected grading criteria, i.e., functionality, code quality, and algorithmic efficiency, ensuring a more comprehensive and transparent evaluation. This interconnectedness necessitates the use of CoT to systematically address each criterion while considering their mutual influence. To empirically validate the efficiency of StepGrade, we conducted a case study involving 30 Python programming assignments across three difficulty levels (easy, intermediate, and advanced). The approach is validated against expert human evaluations to assess its consistency, accuracy, and fairness. Results demonstrate that CoT prompting significantly outperforms regular prompting in both grading quality and interpretability. By reducing the time and effort required for manual grading, this research demonstrates the potential of GPT-4 with CoT prompting to revolutionize programming education through scalable and pedagogically effective automated grading systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20840v1">CodeTool: Enhancing Programmatic Tool Invocation of LLMs via Process Supervision</a></div>
    <div class="paper-meta">
      📅 2025-03-26
    </div>
    <details class="paper-abstract">
      Tool invocation significantly enhances the capabilities of Large Language Models (LLMs), yet challenges persist, particularly in complex task scenarios. Current methods, such as instruction-enhanced reasoning and supervised fine-tuning, often result in unnecessarily long reasoning paths and face difficulties in verifying the correctness of intermediate steps. In this paper, we propose CodeTool, a novel framework for stepwise code generation that improves LLM tool invocation by leveraging the concise and easily verifiable nature of code. CodeTool incorporates two distinct process rewards: the On-the-spot Reward, which provides immediate feedback on the accuracy of each tool invocation, and the Latent Reward, which assesses the contribution of each step toward overall task completion. By maximizing the cumulative reward of the On-the-spot and Latend Rewards at each step, LLMs are guided to follow efficient and accurate reasoning paths. Extensive experiments on StableToolBench and RestBench-TMDB demonstrate the superiority of CodeTool over existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20823v1">Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy</a></div>
    <div class="paper-meta">
      📅 2025-03-26
      | 💬 Accepted at CVPR2025
    </div>
    <details class="paper-abstract">
      Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety alignment. Experiments across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and o1 with high attack success rate, which previous attack approaches have consistently struggled to jailbreak. Code is available at https://github.com/naver-ai/JOOD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19855v1">Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs), such as OpenAI-o1 and DeepSeek-R1, have demonstrated the effectiveness of test-time scaling, where extended reasoning processes substantially enhance model performance. Despite this, current models are constrained by limitations in handling long texts and reinforcement learning (RL) training efficiency. To address these issues, we propose a simple yet effective test-time scaling approach Multi-round Thinking. This method iteratively refines model reasoning by leveraging previous answers as prompts for subsequent rounds. Extensive experiments across multiple models, including QwQ-32B and DeepSeek-R1, consistently show performance improvements on various benchmarks such as AIME 2024, MATH-500, GPQA-diamond, and LiveCodeBench. For instance, the accuracy of QwQ-32B improved from 80.3% (Round 1) to 82.1% (Round 2) on the AIME 2024 dataset, while DeepSeek-R1 showed a similar increase from 79.7% to 82.0%. These results confirm that Multi-round Thinking is a broadly applicable, straightforward approach to achieving stable enhancements in model performance, underscoring its potential for future developments in test-time scaling techniques. The key prompt: {Original question prompt} The assistant's previous answer is: <answer> {last round answer} </answer>, and please re-answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19850v1">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Information retrieval in hour-long videos presents a significant challenge, even for state-of-the-art Vision-Language Models (VLMs), particularly when the desired information is localized within a small subset of frames. Long video data presents challenges for VLMs due to context window limitations and the difficulty of pinpointing frames containing the answer. Our novel video agent, FALCONEye, combines a VLM and a Large Language Model (LLM) to search relevant information along the video, and locate the frames with the answer. FALCONEye novelty relies on 1) the proposed meta-architecture, which is better suited to tackle hour-long videos compared to short video approaches in the state-of-the-art; 2) a new efficient exploration algorithm to locate the information using short clips, captions and answer confidence; and 3) our state-of-the-art VLMs calibration analysis for the answer confidence. Our agent is built over a small-size VLM and a medium-size LLM being accessible to run on standard computational resources. We also release FALCON-Bench, a benchmark to evaluate long (average > 1 hour) Video Answer Search challenges, highlighting the need for open-ended question evaluation. Our experiments show FALCONEye's superior performance than the state-of-the-art in FALCON-Bench, and similar or better performance in related benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02341v2">UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Low-altitude mobility, exemplified by unmanned aerial vehicles (UAVs), has introduced transformative advancements across various domains, like transportation, logistics, and agriculture. Leveraging flexible perspectives and rapid maneuverability, UAVs extend traditional systems' perception and action capabilities, garnering widespread attention from academia and industry. However, current UAV operations primarily depend on human control, with only limited autonomy in simple scenarios, and lack the intelligence and adaptability needed for more complex environments and tasks. The emergence of large language models (LLMs) demonstrates remarkable problem-solving and generalization capabilities, offering a promising pathway for advancing UAV intelligence. This paper explores the integration of LLMs and UAVs, beginning with an overview of UAV systems' fundamental components and functionalities, followed by an overview of the state-of-the-art in LLM technology. Subsequently, it systematically highlights the multimodal data resources available for UAVs, which provide critical support for training and evaluation. Furthermore, it categorizes and analyzes key tasks and application scenarios where UAVs and LLMs converge. Finally, a reference roadmap towards agentic UAVs is proposed, aiming to enable UAVs to achieve agentic intelligence through autonomous perception, memory, reasoning, and tool utilization. Related resources are available at https://github.com/Hub-Tian/UAVs_Meet_LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11102v3">Empowering LLMs to Understand and Generate Complex Vector Graphics</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted by CVPR 2025. Project Page: https://ximinng.github.io/LLM4SVGProject/
    </div>
    <details class="paper-abstract">
      The unprecedented advancements in Large Language Models (LLMs) have profoundly impacted natural language processing but have yet to fully embrace the realm of scalable vector graphics (SVG) generation. While LLMs encode partial knowledge of SVG data from web pages during training, recent findings suggest that semantically ambiguous and tokenized representations within LLMs may result in hallucinations in vector primitive predictions. Additionally, LLM training typically lacks modeling and understanding of the rendering sequence of vector paths, which can lead to occlusion between output vector primitives. In this paper, we present LLM4SVG, an initial yet substantial step toward bridging this gap by enabling LLMs to better understand and generate vector graphics. LLM4SVG facilitates a deeper understanding of SVG components through learnable semantic tokens, which precisely encode these tokens and their corresponding properties to generate semantically aligned SVG outputs. Using a series of learnable semantic tokens, a structured dataset for instruction following is developed to support comprehension and generation across two primary tasks. Our method introduces a modular architecture to existing large language models, integrating semantic tags, vector instruction encoders, fine-tuned commands, and powerful LLMs to tightly combine geometric, appearance, and language information. To overcome the scarcity of SVG-text instruction data, we developed an automated data generation pipeline that collected our SVGX-SFT Dataset, consisting of high-quality human-designed SVGs and 580k SVG instruction following data specifically crafted for LLM training, which facilitated the adoption of the supervised fine-tuning strategy popular in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05215v3">DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 EuroSys 2025'
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) greatly improves model quality for downstream tasks. However, serving many fine-tuned LLMs concurrently is challenging due to the sporadic, bursty, and varying request patterns of different LLMs. To bridge this gap, we present DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by up to 10x while maintaining high model quality. The key insight behind this design is that fine-tuning results in small-magnitude changes to the pre-trained model. By co-designing the serving system with the compression algorithm, DeltaZip achieves 2x to 12x improvement in throughput compared to the state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17662v2">Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved breakthrough progress in many dialogue generation tasks. However, their lack of emotion and fine-grained role awareness limits the model's ability to provide personalized and diverse interactions further. Current methods face high costs in collecting high-quality annotated data for scenarios such as role-playing, and traditional human alignment methods are difficult to deploy due to the inherent diversity of model behavior in role-playing scenarios. Inspired by the alignment of models for safety behaviors through RLHF (Reinforcement Learning from Human Feedback), in this paper, we revisit model role-playing behavior from the perspective of persona alignment and propose a novel annotation-free framework named \textbf{\underline{P}}ersona-Aware \textbf{\underline{C}}ontrastive \textbf{\underline{L}}earning (PCL) to align LLMs' behavior during role-playing, enhancing the model's role consistency. Specifically, we first design a role chain method to encourage the model to self-question based on the role characteristics and dialogue context to adjust personality consistency. Then, we further enhance the model's role-playing strategy through iterative contrastive learning between the use of role characteristics and not. Experiments on both black-box and white-box LLMs show that LLMs equipped with PCL significantly outperform vanilla LLMs under automatic evaluation methods (CharEval \& GPT-4) and human expert evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19693v1">AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19620v1">Optimization through In-Context Learning and Iterative LLM Prompting for Nuclear Engineering Design Problems</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Codes and data are available upon request
    </div>
    <details class="paper-abstract">
      The optimization of nuclear engineering designs, such as nuclear fuel assembly configurations, involves managing competing objectives like reactivity control and power distribution. This study explores the use of Optimization by Prompting, an iterative approach utilizing large language models (LLMs), to address these challenges. The method is straightforward to implement, requiring no hyperparameter tuning or complex mathematical formulations. Optimization problems can be described in plain English, with only an evaluator and a parsing script needed for execution. The in-context learning capabilities of LLMs enable them to understand problem nuances, therefore, they have the potential to surpass traditional metaheuristic optimization methods. This study demonstrates the application of LLMs as optimizers to Boiling Water Reactor (BWR) fuel lattice design, showing the capability of commercial LLMs to achieve superior optimization results compared to traditional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03235v2">Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted in ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19598v1">The Greatest Good Benchmark: Measuring LLMs' Alignment with Utilitarian Moral Dilemmas</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The question of how to make decisions that maximise the well-being of all persons is very relevant to design language models that are beneficial to humanity and free from harm. We introduce the Greatest Good Benchmark to evaluate the moral judgments of LLMs using utilitarian dilemmas. Our analysis across 15 diverse LLMs reveals consistently encoded moral preferences that diverge from established moral theories and lay population moral standards. Most LLMs have a marked preference for impartial beneficence and rejection of instrumental harm. These findings showcase the 'artificial moral compass' of LLMs, offering insights into their moral alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05374v3">Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18710v2">To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The massive computational costs associated with large language model (LLM) pretraining have spurred great interest in reduced-precision floating-point representations to accelerate the process. As a result, the BrainFloat16 (BF16) precision has become the de facto standard for LLM training, with hardware support included in recent generations of accelerators. This trend has gone even further in the latest processors, where FP8 has recently been introduced. However, prior experience with FP16, which was found to be less stable than BF16, raises concerns as to whether FP8, with even fewer bits than FP16, can be a cost-effective option for LLM training. We argue that reduced-precision training schemes must have similar training stability and hyperparameter sensitivities to their higher-precision counterparts in order to be cost-effective. However, we find that currently available methods for FP8 training are not robust enough to allow their use as economical replacements. This prompts us to investigate the stability of reduced-precision LLM training in terms of robustness across random seeds, learning rates, and datasets. To this end, we propose new evaluation techniques and a new metric for quantifying loss landscape sharpness in autoregressive language models. By simulating incremental bit reductions in floating-point representations, we analyze the relationship between representational power and training stability with the intent of aiding future research into the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17056v2">The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG Applications Using an LLM's Internal States</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Detecting hallucinations in large language models (LLMs) is critical for enhancing their reliability and trustworthiness. Most research focuses on hallucinations as deviations from information seen during training. However, the opaque nature of an LLM's parametric knowledge complicates the understanding of why generated texts appear ungrounded: The LLM might not have picked up the necessary knowledge from large and often inaccessible datasets, or the information might have been changed or contradicted during further training. Our focus is on hallucinations involving information not used in training, which we determine by using recency to ensure the information emerged after a cut-off date. This study investigates these hallucinations by detecting them at sentence level using different internal states of various LLMs. We present HalluRAG, a dataset designed to train classifiers on these hallucinations. Depending on the model and quantization, MLPs trained on HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results show that IAVs detect hallucinations as effectively as CEVs and reveal that answerable and unanswerable prompts are encoded differently as separate classifiers for these categories improved accuracy. However, HalluRAG showed some limited generalizability, advocating for more diversity in datasets on hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12355v2">DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted by CVPR 25
    </div>
    <details class="paper-abstract">
      The challenge in LLM-based video understanding lies in preserving visual and semantic information in long videos while maintaining a memory-affordable token count. However, redundancy and correspondence in videos have hindered the performance potential of existing methods. Through statistical learning on current datasets, we observe that redundancy occurs in both repeated and answer-irrelevant frames, and the corresponding frames vary with different questions. This suggests the possibility of adopting dynamic encoding to balance detailed video information preservation with token budget reduction. To this end, we propose a dynamic cooperative network, DynFocus, for memory-efficient video encoding in this paper. Specifically, i) a Dynamic Event Prototype Estimation (DPE) module to dynamically select meaningful frames for question answering; (ii) a Compact Cooperative Encoding (CCE) module that encodes meaningful frames with detailed visual appearance and the remaining frames with sketchy perception separately. We evaluate our method on five publicly available benchmarks, and experimental results consistently demonstrate that our method achieves competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00088v2">T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 EuroSys 2025
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) on edge devices is increasingly important to enhance on-device intelligence. Weight quantization is crucial for reducing the memory footprint of LLMs on devices. However, low-bit LLMs necessitate mixed precision matrix multiplication (mpGEMM) of low precision weights and high precision activations during inference. Existing systems, lacking native support for mpGEMM, resort to dequantize weights for high precision computation. Such an indirect way can lead to a significant inference overhead. In this paper, we introduce T-MAC, an innovative lookup table(LUT)-based method designed for efficient low-bit LLM (i.e., weight-quantized LLM) inference on CPUs. T-MAC directly supports mpGEMM without dequantization, while simultaneously eliminating multiplications and reducing additions required. Specifically, T-MAC transforms the traditional data-type-centric multiplication to bit-wise table lookup, and enables a unified and scalable mpGEMM solution. Our LUT-based kernels scale linearly to the weight bit-width. Evaluated on low-bit Llama and BitNet models, T-MAC demonstrates up to 4x increase in throughput and 70% reduction in energy consumption compared to llama.cpp. For BitNet-b1.58-3B, T-MAC delivers a token generation throughput of 30 tokens/s with a single core and 71 tokens/s with eight cores on M2-Ultra, and 11 tokens/s on lower-end devices like Raspberry Pi 5, which significantly exceeds the adult average reading speed. T-MAC with LUT-based computing paradigm, paves the way for the practical deployment of low-bit LLMs on resource-constrained edge devices without compromising computational efficiency. The system is open-sourced at https://github.com/microsoft/T-MAC .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19470v1">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19449v1">VecTrans: LLM Transformation Framework for Better Auto-vectorization on High-performance CPU</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated great capabilities in code generation, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. Vectorization, a crucial optimization for enhancing code performance, often fails because of the compiler's inability to recognize complex code patterns, which commonly require extensive empirical expertise. LLMs, with their ability to capture intricate patterns, thus providing a promising solution to this challenge. This paper presents VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compiler's auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. Experimental results show that among all 50 TSVC functions unvectorizable by Clang, GCC, and BiShengCompiler, VecTrans successfully vectorizes 23 cases (46%) and achieves an average speedup of 2.02x, greatly surpassing state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00599v3">VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 17 pages, 14 figures, technical report
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have recently exhibited remarkable capabilities in general video understanding. However, they mainly focus on holistic comprehension and struggle with capturing fine-grained spatial and temporal details. Besides, the lack of high-quality object-level video instruction data and a comprehensive benchmark further hinders their advancements. To tackle these challenges, we introduce the VideoRefer Suite to empower Video LLM for finer-level spatial-temporal video understanding, i.e., enabling perception and reasoning on any objects throughout the video. Specially, we thoroughly develop VideoRefer Suite across three essential aspects: dataset, model, and benchmark. Firstly, we introduce a multi-agent data engine to meticulously curate a large-scale, high-quality object-level video instruction dataset, termed VideoRefer-700K. Next, we present the VideoRefer model, which equips a versatile spatial-temporal object encoder to capture precise regional and sequential representations. Finally, we meticulously create a VideoRefer-Bench to comprehensively assess the spatial-temporal understanding capability of a Video LLM, evaluating it across various aspects. Extensive experiments and analyses demonstrate that our VideoRefer model not only achieves promising performance on video referring benchmarks but also facilitates general video understanding capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11960v3">MCRanker: Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The most recent pointwise Large Language Model (LLM) rankers have achieved remarkable ranking results. However, these rankers are hindered by two major drawbacks: (1) they fail to follow a standardized comparison guidance during the ranking process, and (2) they struggle with comprehensive considerations when dealing with complicated passages. To address these shortcomings, we propose to build a ranker that generates ranking scores based on a set of criteria from various perspectives. These criteria are intended to direct each perspective in providing a distinct yet synergistic evaluation. Our research, which examines eight datasets from the BEIR benchmark demonstrates that incorporating this multi-perspective criteria ensemble approach markedly enhanced the performance of pointwise LLM rankers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20021v2">Think Carefully and Check Again! Meta-Generation Unlocking LLMs for Low-Resource Cross-Lingual Summarization</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Cross-lingual summarization (CLS) aims to generate a summary for the source text in a different target language. Currently, instruction-tuned large language models (LLMs) excel at various English tasks. However, unlike languages such as English, Chinese or Spanish, for those relatively low-resource languages with limited usage or data, recent studies have shown that LLMs' performance on CLS tasks remains unsatisfactory even with few-shot settings. This raises the question: Are LLMs capable of handling cross-lingual summarization tasks for low-resource languages? To resolve this question, we fully explore the potential of large language models on cross-lingual summarization task for low-resource languages through our four-step zero-shot method: Summarization, Improvement, Translation and Refinement (SITR) with correspondingly designed prompts. We test our proposed method with multiple LLMs on two well-known cross-lingual summarization datasets with various low-resource target languages. The results show that: i) GPT-3.5 and GPT-4 significantly and consistently outperform other baselines when using our zero-shot SITR methods. ii) By employing our proposed method, we unlock the potential of LLMs, enabling them to effectively handle cross-lingual summarization tasks for relatively low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20016v2">Vulnerability of LLMs to Vertically Aligned Text Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Text classification involves categorizing a given text, such as determining its sentiment or identifying harmful content. With the advancement of large language models (LLMs), these models have become highly effective at performing text classification tasks. However, they still show vulnerabilities to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input? In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) Chain of Thought (CoT) reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but few-shot learning with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19353v1">QUAD: Quantization and Parameter-Efficient Tuning of LLM with Activation Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 18 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in diverse applications but suffer inefficiency due to massive scale. While quantization reduces computational costs, existing methods degrade accuracy in medium-sized LLMs (e.g., Llama-3-8B) due to activation outliers. To address this, we propose QUAD (Quantization with Activation Decomposition), a framework leveraging Singular Value Decomposition (SVD) to suppress activation outliers for effective 4-bit quantization. QUAD estimates activation singular vectors offline using calibration data to construct an orthogonal transformation matrix P, shifting outliers to additional dimensions in full precision while quantizing rest components to 4-bit. Additionally, QUAD enables parameter-efficient fine-tuning via adaptable full-precision outlier weights, narrowing the accuracy gap between quantized and full-precision models. Experiments demonstrate that QUAD achieves 94% ~ 96% accuracy under W4A4 quantization and 98% accuracy with W4A4/A8 and parameter-efficient fine-tuning for Llama-3 and Qwen-2.5 models. Our code is available at \href{https://github.com/hyx1999/Quad}{repository}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19326v1">Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20084v1">Can Multi-modal (reasoning) LLMs work as deepfake detectors?</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Deepfake detection remains a critical challenge in the era of advanced generative models, particularly as synthetic media becomes more sophisticated. In this study, we explore the potential of state of the art multi-modal (reasoning) large language models (LLMs) for deepfake image detection such as (OpenAI O1/4o, Gemini thinking Flash 2, Deepseek Janus, Grok 3, llama 3.2, Qwen 2/2.5 VL, Mistral Pixtral, Claude 3.5/3.7 sonnet) . We benchmark 12 latest multi-modal LLMs against traditional deepfake detection methods across multiple datasets, including recently published real-world deepfake imagery. To enhance performance, we employ prompt tuning and conduct an in-depth analysis of the models' reasoning pathways to identify key contributing factors in their decision-making process. Our findings indicate that best multi-modal LLMs achieve competitive performance with promising generalization ability with zero shot, even surpass traditional deepfake detection pipelines in out-of-distribution datasets while the rest of the LLM families performs extremely disappointing with some worse than random guess. Furthermore, we found newer model version and reasoning capabilities does not contribute to performance in such niche tasks of deepfake detection while model size do help in some cases. This study highlights the potential of integrating multi-modal reasoning in future deepfake detection frameworks and provides insights into model interpretability for robustness in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18316v2">Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20036v1">BugCraft: End-to-End Crash Bug Reproduction Using LLM Agents in Minecraft</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Reproducing game bugs, in our case crash bugs in continuously evolving games like Minecraft, is a notoriously manual, time-consuming, and challenging process to automate. Despite the success of LLM-driven bug reproduction in other software domains, games, with their complex interactive environments, remain largely unaddressed. This paper introduces BugCraft, a novel end-to-end framework designed to automate the reproduction of crash bugs in Minecraft directly from user-submitted bug reports, addressing the critical gap in automated game bug reproduction. BugCraft employs a two-stage approach: first, a Step Synthesizer leverages LLMs and Minecraft Wiki knowledge to transform bug reports into high-quality, structured steps to reproduce (S2R). Second, an Action Model, powered by a vision-based LLM agent (GPT-4o) and a custom macro API, executes these S2R steps within Minecraft to trigger the reported crash. To facilitate evaluation, we introduce BugCraft-Bench, a curated dataset of Minecraft crash bug reports. Evaluated on BugCraft-Bench, our framework successfully reproduced 30.23% of crash bugs end-to-end. The Step Synthesizer demonstrated a 66.28% accuracy in generating correct bug reproduction plans, highlighting its effectiveness in interpreting and structuring bug report information. BugCraft demonstrates the feasibility of automated reproduction of crash bugs in complex game environments using LLMs, opening promising avenues for game testing and development. The framework and the BugCraft-Bench dataset pave the way for future research in automated game bug analysis and hold potential for generalization to other interactive game platforms. Finally, we make our code open at https://bugcraft2025.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19936v1">VisualQuest: A Diverse Image Dataset for Evaluating Visual Recognition in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      This paper introduces VisualQuest, a novel image dataset designed to assess the ability of large language models (LLMs) to interpret non-traditional, stylized imagery. Unlike conventional photographic benchmarks, VisualQuest challenges models with images that incorporate abstract, symbolic, and metaphorical elements, requiring the integration of domain-specific knowledge and advanced reasoning. The dataset was meticulously curated through multiple stages of filtering, annotation, and standardization to ensure high quality and diversity. Our evaluations using several state-of-the-art multimodal LLMs reveal significant performance variations that underscore the importance of both factual background knowledge and inferential capabilities in visual recognition tasks. VisualQuest thus provides a robust and comprehensive benchmark for advancing research in multimodal reasoning and model architecture design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18929v1">Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a critical component of large language model (LLM) post-training. However, existing on-policy algorithms used for post-training are inherently incompatible with the use of experience replay buffers, which can be populated scalably by distributed off-policy actors to enhance exploration as compute increases. We propose efficiently obtaining this benefit of replay buffers via Trajectory Balance with Asynchrony (TBA), a massively scalable LLM RL system. In contrast to existing approaches, TBA uses a larger fraction of compute on search, constantly generating off-policy data for a central replay buffer. A training node simultaneously samples data from this buffer based on reward or recency to update the policy using Trajectory Balance (TB), a diversity-seeking RL objective introduced for GFlowNets. TBA offers three key advantages: (1) decoupled training and search, speeding up training wall-clock time by 4x or more; (2) improved diversity through large-scale off-policy sampling; and (3) scalable search for sparse reward settings. On mathematical reasoning, preference-tuning, and automated red-teaming (diverse and representative post-training tasks), TBA produces speed and performance improvements over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18891v1">AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) based on large language models (LLMs) have demonstrated significant potential in collaborative problem-solving. However, they still face substantial challenges of low communication efficiency and suboptimal task performance, making the careful design of the agents' communication topologies particularly important. Inspired by the management theory that roles in an efficient team are often dynamically adjusted, we propose AgentDropout, which identifies redundant agents and communication across different communication rounds by optimizing the adjacency matrices of the communication graphs and eliminates them to enhance both token efficiency and task performance. Compared to state-of-the-art methods, AgentDropout achieves an average reduction of 21.6% in prompt token consumption and 18.4% in completion token consumption, along with a performance improvement of 1.14 on the tasks. Furthermore, the extended experiments demonstrate that AgentDropout achieves notable domain transferability and structure robustness, revealing its reliability and effectiveness. We release our code at https://github.com/wangzx1219/AgentDropout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18869v1">Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 9 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The efficiency of Large Language Model~(LLM) inference is often constrained by substantial memory bandwidth and capacity demands. Existing techniques, such as pruning, quantization, and mixture of experts/depth, reduce memory capacity and/or bandwidth consumption at the cost of slight degradation in inference quality. This paper introduces a design solution that further alleviates memory bottlenecks by enhancing the on-chip memory controller in AI accelerators to achieve two main objectives: (1) significantly reducing memory capacity and bandwidth usage through lossless block compression~(e.g., LZ4 and ZSTD) of model weights and key-value (KV) cache without compromising inference quality, and (2) enabling memory bandwidth and energy consumption to scale proportionally with context-dependent dynamic quantization. These goals are accomplished by equipping the on-chip memory controller with mechanisms to improve fine-grained bit-level accessibility and compressibility of weights and KV cache through LLM-aware configuration of in-memory placement and representation. Experimental results on publicly available LLMs demonstrate the effectiveness of this approach, showing memory footprint reductions of 25.2\% for model weights and 46.9\% for KV cache. In addition, our hardware prototype at 4\,GHz and 32 lanes (7\,nm) achieves 8\,TB/s throughput with a modest area overhead (under 3.8\,mm\(^2\)), which underscores the viability of LLM-aware memory control as a key to efficient large-scale inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18825v1">EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18809v1">Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have shown remarkable capabilities in various artificial intelligence problems. However, they fail to plan reliably, even when prompted with a detailed definition of the planning task. Attempts to improve their planning capabilities, such as chain-of-thought prompting, fine-tuning, and explicit "reasoning" still yield incorrect plans and usually fail to generalize to larger tasks. In this paper, we show how to use LLMs to generate correct plans, even for out-of-distribution tasks of increasing size. For a given planning domain, we ask an LLM to generate several domain-dependent heuristic functions in the form of Python code, evaluate them on a set of training tasks within a greedy best-first search, and choose the strongest one. The resulting LLM-generated heuristics solve many more unseen test tasks than state-of-the-art domain-independent heuristics for classical planning. They are even competitive with the strongest learning algorithm for domain-dependent planning. These findings are especially remarkable given that our proof-of-concept implementation is based on an unoptimized Python planner and the baselines all build upon highly optimized C++ code. In some domains, the LLM-generated heuristics expand fewer states than the baselines, revealing that they are not only efficiently computable, but sometimes even more informative than the state-of-the-art heuristics. Overall, our results show that sampling a set of planning heuristic function programs can significantly improve the planning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18792v1">REALM: A Dataset of Real-World LLM Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models, such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles. A dedicated dashboard https://realm-e7682.web.app/ presents the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18773v1">BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The growing adoption of long-context Large Language Models (LLMs) has introduced significant memory and computational challenges in autoregressive decoding due to the expanding Key-Value (KV) cache. KV cache quantization has emerged as a promising solution, with prior work showing that 4-bit or even 2-bit quantization can maintain model accuracy while reducing memory costs. However, despite these benefits, preliminary implementations for the low-bit KV cache struggle to deliver the expected speedup due to quantization and dequantization overheads and the lack of Tensor Cores utilization. In this work, we propose BitDecoding, a GPU-optimized framework that unlocks Tensor Cores for efficient decoding with low-bit KV cache. Efficiently leveraging Tensor Cores for low-bit KV cache is challenging due to the dynamic nature of KV cache generation at each decoding step. BitDecoding addresses these challenges with a Tensor Cores-Centric BitFusion Scheme that ensures data layout compatibility to enable high utilization of Tensor Cores. Additionally, BitDecoding incorporates a warp-efficient parallel decoding kernel and a fine-grained asynchronous pipeline, minimizing dequantization overhead and improving computational efficiency. Experiments show that BitDecoding achieves up to 7.5x speedup on RTX 4090, 4.8x on A100, and 8.9x on H100, compared to FP16 FlashDecoding-v2. It also outperforms the state-of-the-art low-bit KV cache implementation (QServe) by up to 4.3x. On LLaMA-3.1-8B with a 128K sequence length, BitDecoding reduces single-batch decoding latency by 3x, demonstrating its effectiveness in long-context generation scenarios. The code is available at https://github.com/DD-DuDa/BitDecoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16514v2">VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v2">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.We conduct an repository for our benchmark at https://github.com/zjq0455/PTQ_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18666v1">AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identifying 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v3">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05843v3">From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Our key innovation lies in bridging the semantic gap between object detection and event understanding without requiring expensive task-specific training. The proposed plug-and-play framework interfaces with any open-vocabulary detector while extending their inherent capabilities across architectures. At its core, our approach combines (i) a symbolic regression mechanism exploring relationship patterns among detected entities and (ii) a LLM-guided strategically guiding the search toward meaningful expressions. These discovered symbolic rules transform low-level visual perception into interpretable event understanding, providing a transparent reasoning path from objects to events with strong transferability across domains.We compared our training-free framework against specialized event recognition systems across diverse application domains. Experiments demonstrate that our framework enhances multiple object detector architectures to recognize complex events such as illegal fishing activities (75% AUROC, +8.36% improvement), construction safety violations (+15.77%), and abnormal crowd behaviors (+23.16%). The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18599v1">Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 15 pages, 14 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Model serving system batches multiple requests to achieve high throughput, while batching attention operations is challenging, rendering memory bandwidth a critical bottleneck. The community relies on high-end GPUs with multiple high-bandwidth memory channels. Unfortunately, HBM's high bandwidth often comes at the expense of limited memory capacity, which reduces core utilization and increases costs. Recent advancements enabling longer contexts for LLMs have substantially increased the key-value cache size, further intensifying the pressures on memory capacity. The literature has explored KV cache quantization techniques, which commonly use low bitwidth for most values, selectively using higher bitwidth for outlier values. While this approach helps achieve high accuracy and low bitwidth simultaneously, it comes with the limitation that cost for online outlier detection is excessively high, negating the advantages. We propose Oaken, an acceleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware. To effectively find a sweet spot in the accuracy-performance trade-off space of KV cache quantization, Oaken employs an online-offline hybrid approach, setting outlier thresholds offline, which are then used to determine the quantization scale online. To translate the proposed algorithmic technique into tangible performance gains, Oaken also comes with custom quantization engines and memory management units that can be integrated with any LLM accelerators. We built an Oaken accelerator on top of an LLM accelerator, LPU, and conducted a comprehensive evaluation. Our experiments show that for a batch size of 256, Oaken achieves up to 1.58x throughput improvement over NVIDIA A100 GPU, incurring a minimal accuracy loss of only 0.54\% on average, compared to state-of-the-art KV cache quantization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11702v3">Toward a method for LLM-enabled Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 50.54% correct indications and a maximum of 77.78%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21321v2">LLM Post-Training: A Deep Dive into Reasoning Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 32 pages, 7 figures, 3 tables, 377 references. Github Repo: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed the natural language processing landscape and brought to life diverse applications. Pretraining on vast web-scale data has laid the foundation for these models, yet the research community is now increasingly shifting focus toward post-training techniques to achieve further breakthroughs. While pretraining provides a broad linguistic foundation, post-training methods enable LLMs to refine their knowledge, improve reasoning, enhance factual accuracy, and align more effectively with user intents and ethical considerations. Fine-tuning, reinforcement learning, and test-time scaling have emerged as critical strategies for optimizing LLMs performance, ensuring robustness, and improving adaptability across various real-world tasks. This survey provides a systematic exploration of post-training methodologies, analyzing their role in refining LLMs beyond pretraining, addressing key challenges such as catastrophic forgetting, reward hacking, and inference-time trade-offs. We highlight emerging directions in model alignment, scalable adaptation, and inference-time reasoning, and outline future research directions. We also provide a public repository to continually track developments in this fast-evolving field: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18432v1">Teaching LLMs for Step-Level Automatic Math Correction via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Automatic math correction aims to check students' solutions to mathematical problems via artificial intelligence technologies. Most existing studies focus on judging the final answer at the problem level, while they ignore detailed feedback on each step in a math problem-solving process, which requires abilities of semantic understanding and reasoning. In this paper, we propose a reinforcement learning (RL)-based method to boost large language model (LLM) for step-level automatic math correction, named StepAMC. Particularly, we convert the step-level automatic math correction within the text classification task into an RL problem to enhance the reasoning capabilities of LLMs. Then, we design a space-constrained policy network to improve the stability of RL. Then, we introduce a fine-grained reward network to convert the binary human feedback into a continuous value. We conduct extensive experiments over two benchmark datasets and the results show that our model outperforms the eleven strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17174v3">CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Causal Significance and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 6 pages,4 figures. This paper has been accepted for presentation at IEEE International Conference on Multimedia & Expo 2025
    </div>
    <details class="paper-abstract">
      Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal hallucinations between a step of reasoning and corresponding state transitions are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18377v1">Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities, but their enormous size poses significant challenges for deployment in real-world applications. To address this issue, researchers have sought to apply network pruning techniques to LLMs. A critical challenge in pruning is allocation the sparsity for each layer. Recent sparsity allocation methods is often based on heuristics or search that can easily lead to suboptimal performance. In this paper, we conducted an extensive investigation into various LLMs and revealed three significant discoveries: (1) the layerwise pruning sensitivity (LPS) of LLMs is highly non-uniform, (2) the choice of pruning metric affects LPS, and (3) the performance of a sparse model is related to the uniformity of its layerwise redundancy level. Based on these observations, we propose that the layerwise sparsity of LLMs should adhere to three principles: \emph{non-uniformity}, \emph{pruning metric dependency}, and \emph{uniform layerwise redundancy level} in the pruned model. To this end, we proposed Maximum Redundancy Pruning (MRP), an iterative pruning algorithm that prunes in the most redundant layers (\emph{i.e.}, those with the highest non-outlier ratio) at each iteration. The achieved layerwise sparsity aligns with the outlined principles. We conducted extensive experiments on publicly available LLMs, including the LLaMA2 and OPT, across various benchmarks. Experimental results validate the effectiveness of MRP, demonstrating its superiority over previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v3">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications.
    </details>
</div>
