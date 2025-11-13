# llm - 2024_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00655v1">Finding Missed Code Size Optimizations in Compilers using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 Accepted to appear in The International Conference on Compiler Construction (CC) 2025
    </div>
    <details class="paper-abstract">
      Compilers are complex, and significant effort has been expended on testing them. Techniques such as random program generation and differential testing have proved highly effective and have uncovered thousands of bugs in production compilers. The majority of effort has been expended on validating that a compiler produces correct code for a given input, while less attention has been paid to ensuring that the compiler produces performant code. In this work we adapt differential testing to the task of identifying missed optimization opportunities in compilers. We develop a novel testing approach which combines large language models (LLMs) with a series of differential testing strategies and use them to find missing code size optimizations in C / C++ compilers. The advantage of our approach is its simplicity. We offload the complex task of generating random code to an off-the-shelf LLM, and use heuristics and analyses to identify anomalous compiler behavior. Our approach requires fewer than 150 lines of code to implement. This simplicity makes it extensible. By simply changing the target compiler and initial LLM prompt we port the approach from C / C++ to Rust and Swift, finding bugs in both. To date we have reported 24 confirmed bugs in production compilers, and conclude that LLM-assisted testing is a promising avenue for detecting optimization bugs in real world compilers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00581v1">Causal Graph Guided Steering of LLM Values via Prompts and Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into critical applications, aligning their behavior with human values presents significant challenges. Current methods, such as Reinforcement Learning from Human Feedback (RLHF), often focus on a limited set of values and can be resource-intensive. Furthermore, the correlation between values has been largely overlooked and remains underutilized. Our framework addresses this limitation by mining a causal graph that elucidates the implicit relationships among various values within the LLMs. Leveraging the causal graph, we implement two lightweight mechanisms for value steering: prompt template steering and Sparse Autoencoder feature steering, and analyze the effects of altering one value dimension on others. Extensive experiments conducted on Gemma-2B-IT and Llama3-8B-IT demonstrate the effectiveness and controllability of our steering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00560v1">Re-evaluating Automatic LLM System Ranking for Alignment with Human Preference</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Evaluating and ranking the capabilities of different LLMs is crucial for understanding their performance and alignment with human preferences. Due to the high cost and time-consuming nature of human evaluations, an automatic LLM bencher (i.e., an automatic evaluation framework that aims to rank LLMs based on their alignment with human preferences) is indispensable. An automatic LLM bencher consists of four components: the input set (e.g., a user instruction), the evaluation model (e.g., an LLM), the evaluation type (e.g., pairwise comparison), and the aggregation method (e.g., the ELO rating system). However, previous work has not thoroughly explored how to select these components or how their different combinations influence the results. In this work, through controlled experiments, we provide a series of recommendations on how to choose each component to better automate the evaluation of LLMs. Furthermore, we discovered that when evaluating LLMs with similar performance, the performance of the automatic LLM bencher declines sharply, underscoring the limitations of current benchers and calling for future work. Lastly, we found that the evaluation models' performance at the instance level (e.g., the accuracy of selecting the best output) does not always align with their effectiveness when used as a component of a bencher, highlighting the importance of dedicated system-level evaluation of benchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00559v1">AraSTEM: A Native Arabic Multiple Choice Question Benchmark for Evaluating LLMs Knowledge In STEM Subjects</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities, not only in generating human-like text, but also in acquiring knowledge. This highlights the need to go beyond the typical Natural Language Processing downstream benchmarks and asses the various aspects of LLMs including knowledge and reasoning. Numerous benchmarks have been developed to evaluate LLMs knowledge, but they predominantly focus on the English language. Given that many LLMs are multilingual, relying solely on benchmarking English knowledge is insufficient. To address this issue, we introduce AraSTEM, a new Arabic multiple-choice question dataset aimed at evaluating LLMs knowledge in STEM subjects. The dataset spans a range of topics at different levels which requires models to demonstrate a deep understanding of scientific Arabic in order to achieve high accuracy. Our findings show that publicly available models of varying sizes struggle with this dataset, and underscores the need for more localized language models. The dataset is freely accessible on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00555v1">Monty Hall and Optimized Conformal Prediction to Improve Decision-Making with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are empowering decision-making in several applications, including tool or API usage and answering multiple-choice questions (MCQs). However, they often make overconfident, incorrect predictions, which can be risky in high-stakes settings like healthcare and finance. To mitigate these risks, recent works have used conformal prediction (CP), a model-agnostic framework for distribution-free uncertainty quantification. CP transforms a \emph{score function} into prediction sets that contain the true answer with high probability. While CP provides this coverage guarantee for arbitrary scores, the score quality significantly impacts prediction set sizes. Prior works have relied on LLM logits or other heuristic scores, lacking quality guarantees. We address this limitation by introducing CP-OPT, an optimization framework to learn scores that minimize set sizes while maintaining coverage. Furthermore, inspired by the Monty Hall problem, we extend CP's utility beyond uncertainty quantification to improve accuracy. We propose \emph{conformal revision of questions} (CROQ) to revise the problem by narrowing down the available choices to those in the prediction set. The coverage guarantee of CP ensures that the correct choice is in the revised question prompt with high probability, while the smaller number of choices increases the LLM's chances of answering it correctly. Experiments on MMLU, ToolAlpaca, and TruthfulQA datasets with Gemma-2, Llama-3 and Phi-3 models show that CP-OPT significantly reduces set sizes while maintaining coverage, and CROQ improves accuracy over the standard inference, especially when paired with CP-OPT scores. Together, CP-OPT and CROQ offer a robust framework for improving both the safety and accuracy of LLM-driven decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16928v2">From Sands to Mansions: Simulating Full Attack Chain with LLM-Organized Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Adversarial dynamics are intrinsic to the nature of offense and defense in cyberspace, with both attackers and defenders continuously evolving their technologies. Given the wide array of security products available, users often face challenges in selecting the most effective solutions. Furthermore, traditional benchmarks based on single-point attacks are increasingly inadequate, failing to accurately reflect the full range of attacker capabilities and falling short in properly evaluating the effectiveness of defense products. Automated multi-stage attack simulations offer a promising approach to enhance system evaluation efficiency and aid in analyzing the effectiveness of detection systems. However, simulating a full attack chain is complex and requires significant time and expertise from security professionals, facing several challenges, including limited coverage of attack techniques, a high level of required expertise, and a lack of execution detail. In this paper, we model automatic attack simulation as a planning problem. By using the Planning Domain Definition Language (PDDL) to formally describe the attack simulation problem, and combining domain knowledge of both the problem and the domain space, we enable the planning of attack paths through standardized, domain-independent planning algorithms. We explore the potential of Large Language Models (LLMs) to summarize and analyze knowledge from existing attack documentation and reports, facilitating automated attack planning. We introduce Aurora, a system that autonomously simulates full attack chains based on external attack tools and threat intelligence reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00363v1">SPDZCoder: Teaching LLMs to Synthesize Privacy Computing Code without Massive Training Data</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Privacy computing receives increasing attention but writing privacy computing code remains challenging for developers due to limited library functions that necessitate extensive function implementation from scratch as well as the data-oblivious requirement which contradicts intuitive thinking and usual practices of programmers. Large language models (LLMs) have demonstrated surprising capabilities in coding tasks and achieved state-of-the-art performance across many benchmarks. However, even with extensive prompting, existing LLMs struggle with code translation task for privacy computing, such as translating Python to MP-SPDZ, due to the scarcity of MP-SPDZ data required for effective pre-training or fine-tuning. To address the limitation, this paper proposes SPDZCoder, a rule-based framework to teach LLMs to synthesize privacy computing code without asking experts to write tons of code and by leveraging the instruction-following and in-context learning ability of LLMs. Specifically, SPDZCoder decouples the translation task into the refactoring stage and the generation stage, which can mitigate the semantic-expressing differences at different levels. In addition, SPDZCoder can further improve its performance by a feedback stage. SPDZCoder does not require fine-tuning since it adopts an in-context learning paradigm of LLMs. To evaluate SPDZCoder, we manually created a benchmark dataset, named SPDZEval, containing six classes of difficult tasks to implement in MP-SPDZ. We conduct experiments on SPDZEval and the experimental results shows that SPDZCoder achieves the state-of-the-art performance in pass@1 and pass@2 across six data splits. Specifically, SPDZCoder achieves an overall correctness of 85.94% and 92.01% in pass@1 and pass@2, respectively, significantly surpassing baselines (at most 30.35% and 49.84% in pass@1 and pass@2, respectively) by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.15453v2">Benchmarking the Performance of Pre-trained LLMs across Urdu NLP Tasks</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) pre-trained on multilingual data have revolutionized natural language processing research, by transitioning from languages and task specific model pipelines to a single model adapted on a variety of tasks. However majority of existing multilingual NLP benchmarks for LLMs provide evaluation data in only few languages with little linguistic diversity. In addition these benchmarks lack quality assessment against the respective state-of the art models. This study presents an in-depth examination of 7 prominent LLMs: GPT-3.5-turbo, Llama 2-7B-Chat, Llama 3.1-8B, Bloomz 3B, Bloomz 7B1, Ministral-8B and Whisper (Large, medium and small variant) across 17 tasks using 22 datasets, 13.8 hours of speech, in a zero-shot setting, and their performance against state-of-the-art (SOTA) models, has been compared and analyzed. Our experiments show that SOTA models currently outperform encoder-decoder models in majority of Urdu NLP tasks under zero-shot settings. However, comparing Llama 3.1-8B over prior version Llama 2-7B-Chat, we can deduce that with improved language coverage, LLMs can surpass these SOTA models. Our results emphasize that models with fewer parameters but richer language-specific data, like Llama 3.1-8B, often outperform larger models with lower language diversity, such as GPT-3.5, in several tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00353v1">RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a key paradigm for enhancing large language models (LLMs) by incorporating external knowledge. However, current RAG methods face two limitations: (1) they only cover limited RAG scenarios. (2) They suffer from limited task diversity due to the lack of a general RAG dataset. To address these limitations, we propose RAG-Instruct, a general method for synthesizing diverse and high-quality RAG instruction data based on any source corpus. Our approach leverages (1) five RAG paradigms, which encompass diverse query-document relationships, and (2) instruction simulation, which enhances instruction diversity and quality by utilizing the strengths of existing instruction datasets. Using this method, we construct a 40K instruction dataset from Wikipedia, comprehensively covering diverse RAG scenarios and tasks. Experiments demonstrate that RAG-Instruct effectively enhances LLMs' RAG capabilities, achieving strong zero-shot performance and significantly outperforming various RAG baselines across a diverse set of tasks. RAG-Instruct is publicly available at https://github.com/FreedomIntelligence/RAG-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10516v3">RetrievalAttention: Accelerating Long-Context LLM Inference via Vector Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Transformer-based Large Language Models (LLMs) have become increasingly important. However, due to the quadratic time complexity of attention computation, scaling LLMs to longer contexts incurs extremely slow inference speed and high GPU memory consumption for caching key-value (KV) vectors. This paper proposes RetrievalAttention, a training-free approach to both accelerate attention computation and reduce GPU memory consumption. By leveraging the dynamic sparsity of attention mechanism, RetrievalAttention proposes to build approximate nearest neighbor search (ANNS) indexes for KV vectors in CPU memory and retrieve the most relevant ones through vector search during generation. Unfortunately, we observe that the off-the-shelf ANNS indexes are often ineffective for such retrieval tasks due to the out-of-distribution (OOD) between query vectors and key vectors in the attention mechanism. RetrievalAttention addresses the OOD challenge by designing an attention-aware vector search algorithm that can adapt to the distribution of query vectors. Our evaluation demonstrates that RetrievalAttention achieves near full attention accuracy while only requiring access to 1--3% of the data. This leads to a significant reduction in the inference cost of long-context LLMs, with a much lower GPU memory footprint. In particular, RetrievalAttention only needs a single NVIDIA RTX4090 (24GB) to serve 128K tokens for LLMs with 8B parameters, which is capable of generating one token in 0.188 seconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18547v3">Token-Budget-Aware LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Reasoning is critical for large language models (LLMs) to excel in a wide range of tasks. While methods like Chain-of-Thought (CoT) reasoning enhance LLM performance by decomposing problems into intermediate steps, they also incur significant overhead in token usage, leading to increased costs. We find that the reasoning process of current LLMs is unnecessarily lengthy and it can be compressed by including a reasonable token budget in the prompt, but the choice of token budget plays a crucial role in the actual compression effectiveness. We then propose a token-budget-aware LLM reasoning framework, which dynamically estimates token budgets for different problems based on reasoning complexity and uses the estimated token budgets to guide the reasoning process. Experiments show that our method effectively reduces token costs in CoT reasoning with only a slight performance reduction, offering a practical solution to balance efficiency and accuracy in LLM reasoning. Code: https://github.com/GeniusHTX/TALE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00273v1">Echoes in AI: Quantifying Lack of Plot Diversity in LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      With rapid advances in large language models (LLMs), there has been an increasing application of LLMs in creative content ideation and generation. A critical question emerges: can current LLMs provide ideas that are diverse enough to truly bolster the collective creativity? We examine two state-of-the-art LLMs, GPT-4 and LLaMA-3, on story generation and discover that LLM-generated stories often consist of plot elements that are echoed across a number of generations. To quantify this phenomenon, we introduce the Sui Generis score, which estimates how unlikely a plot element is to appear in alternative storylines generated by the same LLM. Evaluating on 100 short stories, we find that LLM-generated stories often contain combinations of idiosyncratic plot elements echoed frequently across generations, while the original human-written stories are rarely recreated or even echoed in pieces. Moreover, our human evaluation shows that the ranking of Sui Generis scores among story segments correlates moderately with human judgment of surprise level, even though score computation is completely automatic without relying on human judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01334v2">Augmenting NER Datasets with LLMs: Towards Automated and Refined Annotation</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      In the field of Natural Language Processing (NLP), Named Entity Recognition (NER) is recognized as a critical technology, employed across a wide array of applications. Traditional methodologies for annotating datasets for NER models are challenged by high costs and variations in dataset quality. This research introduces a novel hybrid annotation approach that synergizes human effort with the capabilities of Large Language Models (LLMs). This approach not only aims to ameliorate the noise inherent in manual annotations, such as omissions, thereby enhancing the performance of NER models, but also achieves this in a cost-effective manner. Additionally, by employing a label mixing strategy, it addresses the issue of class imbalance encountered in LLM-based annotations. Through an analysis across multiple datasets, this method has been consistently shown to provide superior performance compared to traditional annotation methods, even under constrained budget conditions. This study illuminates the potential of leveraging LLMs to improve dataset quality, introduces a novel technique to mitigate class imbalances, and demonstrates the feasibility of achieving high-performance NER in a cost-effective way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00257v1">EQUATOR: A Deterministic Framework for Evaluating LLM Reasoning with Open-Ended Questions. # v1.0.0-beta</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Despite the remarkable coherence of Large Language Models (LLMs), existing evaluation methods often suffer from fluency bias and rely heavily on multiple-choice formats, making it difficult to assess factual accuracy and complex reasoning effectively. LLMs thus frequently generate factually inaccurate responses, especially in complex reasoning tasks, highlighting two prominent challenges: (1) the inadequacy of existing methods to evaluate reasoning and factual accuracy effectively, and (2) the reliance on human evaluators for nuanced judgment, as illustrated by Williams and Huckle (2024)[1], who found manual grading indispensable despite automated grading advancements. To address evaluation gaps in open-ended reasoning tasks, we introduce the EQUATOR Evaluator (Evaluation of Question Answering Thoroughness in Open-ended Reasoning). This framework combines deterministic scoring with a focus on factual accuracy and robust reasoning assessment. Using a vector database, EQUATOR pairs open-ended questions with human-evaluated answers, enabling more precise and scalable evaluations. In practice, EQUATOR significantly reduces reliance on human evaluators for scoring and improves scalability compared to Williams and Huckle's (2004)[1] methods. Our results demonstrate that this framework significantly outperforms traditional multiple-choice evaluations while maintaining high accuracy standards. Additionally, we introduce an automated evaluation process leveraging smaller, locally hosted LLMs. We used LLaMA 3.2B, running on the Ollama binaries to streamline our assessments. This work establishes a new paradigm for evaluating LLM performance, emphasizing factual accuracy and reasoning ability, and provides a robust methodological foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20072v2">Extract Information from Hybrid Long Documents Leveraging LLMs: A Framework and Dataset</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate exceptional performance in textual understanding and tabular reasoning tasks. However, their ability to comprehend and analyze hybrid text, containing textual and tabular data, remains unexplored. The hybrid text often appears in the form of hybrid long documents (HLDs), which far exceed the token limit of LLMs. Consequently, we apply an Automated Information Extraction framework (AIE) to enable LLMs to process the HLDs and carry out experiments to analyse four important aspects of information extraction from HLDs. Given the findings: 1) The effective way to select and summarize the useful part of a HLD. 2) An easy table serialization way is enough for LLMs to understand tables. 3) The naive AIE has adaptability in many complex scenarios. 4) The useful prompt engineering to enhance LLMs on HLDs. To address the issue of dataset scarcity in HLDs and support future work, we also propose the Financial Reports Numerical Extraction (FINE) dataset. The dataset and code are publicly available in the attachments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00223v1">CancerKG.ORG A Web-scale, Interactive, Verifiable Knowledge Graph-LLM Hybrid for Assisting with Optimal Cancer Treatment and Care</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      Here, we describe one of the first Web-scale hybrid Knowledge Graph (KG)-Large Language Model (LLM), populated with the latest peer-reviewed medical knowledge on colorectal Cancer. It is currently being evaluated to assist with both medical research and clinical information retrieval tasks at Moffitt Cancer Center, which is one of the top Cancer centers in the U.S. and in the world. Our hybrid is remarkable as it serves the user needs better than just an LLM, KG or a search-engine in isolation. LLMs as is are known to exhibit hallucinations and catastrophic forgetting as well as are trained on outdated corpora. The state of the art KGs, such as PrimeKG, cBioPortal, ChEMBL, NCBI, and other require manual curation, hence are quickly getting stale. CancerKG is unsupervised and is capable of automatically ingesting and organizing the latest medical findings. To alleviate the LLMs shortcomings, the verified KG serves as a Retrieval Augmented Generation (RAG) guardrail. CancerKG exhibits 5 different advanced user interfaces, each tailored to serve different data modalities better and more convenient for the user.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00217v1">The Potential of LLMs in Automating Software Testing: From Generation to Reporting</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 6 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Having a high quality software is essential in software engineering, which requires robust validation and verification processes during testing activities. Manual testing, while effective, can be time consuming and costly, leading to an increased demand for automated methods. Recent advancements in Large Language Models (LLMs) have significantly influenced software engineering, particularly in areas like requirements analysis, test automation, and debugging. This paper explores an agent-oriented approach to automated software testing, using LLMs to reduce human intervention and enhance testing efficiency. The proposed framework integrates LLMs to generate unit tests, visualize call graphs, and automate test execution and reporting. Evaluations across multiple applications in Python and Java demonstrate the system's high test coverage and efficient operation. This research underscores the potential of LLM-powered agents to streamline software testing workflows while addressing challenges in scalability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11694v4">Enhancing LLM Reasoning with Reward-guided Tree Search</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 Technical Report on Slow Thinking with LLMs: I
    </div>
    <details class="paper-abstract">
      Recently, test-time scaling has garnered significant attention from the research community, largely due to the substantial advancements of the o1 model released by OpenAI. By allocating more computational resources during the inference phase, large language models~(LLMs) can extensively explore the solution space by generating more thought tokens or diverse solutions, thereby producing more accurate responses. However, developing an o1-like reasoning approach is challenging, and researchers have been making various attempts to advance this open area of research. In this paper, we present a preliminary exploration into enhancing the reasoning abilities of LLMs through reward-guided tree search algorithms. This framework is implemented by integrating the policy model, reward model, and search algorithm. It is primarily constructed around a tree search algorithm, where the policy model navigates a dynamically expanding tree guided by a specially trained reward model. The implemented framework is denoted as \textbf{STILL-1}. We thoroughly explore various design considerations necessary for implementing this framework and provide a detailed report of the technical aspects. To assess the effectiveness of our approach, we focus on mathematical reasoning tasks and conduct extensive evaluations on four challenging datasets, significantly enhancing the reasoning abilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2407.03963v2">LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-31
    </div>
    <details class="paper-abstract">
      This paper introduces LLM-jp, a cross-organizational project for the research and development of Japanese large language models (LLMs). LLM-jp aims to develop open-source and strong Japanese LLMs, and as of this writing, more than 1,500 participants from academia and industry are working together for this purpose. This paper presents the background of the establishment of LLM-jp, summaries of its activities, and technical reports on the LLMs developed by LLM-jp. For the latest activities, visit https://llm-jp.nii.ac.jp/en/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.14285v3">LLM-Personalize: Aligning LLM Planners with Human Preferences via Reinforced Self-Training for Housekeeping Robots</a></div>
    <div class="paper-meta">
      📅 2024-12-31
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown significant potential for robotics applications, particularly task planning, by harnessing their language comprehension and text generation capabilities. However, in applications such as household robotics, a critical gap remains in the personalization of these models to individual user preferences. We introduce LLM-Personalize, a novel framework with an optimization pipeline designed to personalize LLM planners for household robotics. Our LLM-Personalize framework features an LLM planner that performs iterative planning in multi-room, partially-observable household scenarios, making use of a scene graph constructed with local observations. The generated plan consists of a sequence of high-level actions which are subsequently executed by a controller. Central to our approach is the optimization pipeline, which combines imitation learning and iterative self-training to personalize the LLM planner. In particular, the imitation learning phase performs initial LLM alignment from demonstrations, and bootstraps the model to facilitate effective iterative self-training, which further explores and aligns the model to user preferences. We evaluate LLM-Personalize on Housekeep, a challenging simulated real-world 3D benchmark for household rearrangements, and show that LLM-Personalize achieves more than a 30 percent increase in success rate over existing LLM planners, showcasing significantly improved alignment with human preferences. Project page: https://gdg94.github.io/projectllmpersonalize/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18566v2">Zero-resource Speech Translation and Recognition with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 ICASSP 2025, 5 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Despite recent advancements in speech processing, zero-resource speech translation (ST) and automatic speech recognition (ASR) remain challenging problems. In this work, we propose to leverage a multilingual Large Language Model (LLM) to perform ST and ASR in languages for which the model has never seen paired audio-text data. We achieve this by using a pre-trained multilingual speech encoder, a multilingual LLM, and a lightweight adaptation module that maps the audio representations to the token embedding space of the LLM. We perform several experiments both in ST and ASR to understand how to best train the model and what data has the most impact on performance in previously unseen languages. In ST, our best model is capable to achieve BLEU scores over 23 in CoVoST2 for two previously unseen languages, while in ASR, we achieve WERs of up to 28.2\%. We finally show that the performance of our system is bounded by the ability of the LLM to output text in the desired language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21102v1">Exploring and Controlling Diversity in LLM-Agent Conversation</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 Accepted for the AAAI 2025 Workshop on Advancing LLM-Based Multi-Agent Collaboration
    </div>
    <details class="paper-abstract">
      Diversity is a critical aspect of multi-agent communication. In this paper, we focus on controlling and exploring diversity in the context of open-domain multi-agent conversations, particularly for world simulation applications. We propose Adaptive Prompt Pruning (APP), a novel method that dynamically adjusts the content of the utterance generation prompt to control diversity using a single parameter, lambda. Through extensive experiments, we show that APP effectively controls the output diversity across models and datasets, with pruning more information leading to more diverse output. We comprehensively analyze the relationship between prompt content and conversational diversity. Our findings reveal that information from all components of the prompt generally constrains the diversity of the output, with the Memory block exerting the most significant influence. APP is compatible with established techniques like temperature sampling and top-p sampling, providing a versatile tool for diversity management. To address the trade-offs of increased diversity, such as inconsistencies with omitted information, we incorporate a post-generation correction step, which effectively balances diversity enhancement with output consistency. Additionally, we examine how prompt structure, including component order and length, impacts diversity. This study addresses key questions surrounding diversity in multi-agent world simulation, offering insights into its control, influencing factors, and associated trade-offs. Our contributions lay the foundation for systematically engineering diversity in LLM-based multi-agent collaborations, advancing their effectiveness in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09807v2">LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      Multiple Choice Question Answering (MCQA) is an important problem with numerous real-world applications, such as medicine, law, and education. The high cost of building MCQA datasets makes few-shot learning pivotal in this domain. While Large Language Models (LLMs) can enable few-shot learning, their direct application in real-world scenarios is often hindered by their high computational cost. To address this challenge, we propose a simple yet effective approach that uses LLMs for data generation and scoring. Our approach utilizes LLMs to create MCQA data which contains questions and choices, and to assign probability scores to the generated choices. We then use the generated data and LLM-assigned scores to finetune a smaller and more efficient encoder-only model, DeBERTa-v3-base by leveraging distillation loss. Extensive experiments on the Massive Multitask Language Understanding (MMLU) benchmark demonstrate that our method improves accuracy from 28.9% to 39.3%, representing a gain of over 10% compared to a baseline finetuned directly on 5-shot examples. This shows the effectiveness of LLM-driven data generation and knowledge distillation for few-shot MCQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21033v1">Plancraft: an evaluation dataset for planning with LLM agents</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21016v1">Automated Robustness Testing for LLM-based NLP Software</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      Benefiting from the advancements in LLMs, NLP software has undergone rapid development. Such software is widely employed in various safety-critical tasks, such as financial sentiment analysis, toxic content moderation, and log generation. To our knowledge, there are no known automated robustness testing methods specifically designed for LLM-based NLP software. Given the complexity of LLMs and the unpredictability of real-world inputs (including prompts and examples), it is essential to examine the robustness of overall inputs to ensure the safety of such software. To this end, this paper introduces the first AutOmated Robustness Testing frAmework, AORTA, which reconceptualizes the testing process into a combinatorial optimization problem. Existing testing methods designed for DNN-based software can be applied to LLM-based software by AORTA, but their effectiveness is limited. To address this, we propose a novel testing method for LLM-based software within AORTA called Adaptive Beam Search. ABS is tailored for the expansive feature space of LLMs and improves testing effectiveness through an adaptive beam width and the capability for backtracking. We successfully embed 18 test methods in the designed framework AORTA and compared the test validity of ABS with three datasets and five threat models. ABS facilitates a more comprehensive and accurate robustness assessment before software deployment, with an average test success rate of 86.138%. Compared to the currently best-performing baseline PWWS, ABS significantly reduces the computational overhead by up to 3441.895 seconds per successful test case and decreases the number of queries by 218.762 times on average. Furthermore, test cases generated by ABS exhibit greater naturalness and transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19312v2">From Interests to Insights: An LLM Approach to Course Recommendations Using Natural Language Queries</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 17 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Most universities in the United States encourage their students to explore academic areas before declaring a major and to acquire academic breadth by satisfying a variety of requirements. Each term, students must choose among many thousands of offerings, spanning dozens of subject areas, a handful of courses to take. The curricular environment is also dynamic, and poor communication and search functions on campus can limit a student's ability to discover new courses of interest. To support both students and their advisers in such a setting, we explore a novel Large Language Model (LLM) course recommendation system that applies a Retrieval Augmented Generation (RAG) method to the corpus of course descriptions. The system first generates an 'ideal' course description based on the user's query. This description is converted into a search vector using embeddings, which is then used to find actual courses with similar content by comparing embedding similarities. We describe the method and assess the quality and fairness of some example prompts. Steps to deploy a pilot system on campus are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20993v1">Efficiently Serving LLM Reasoning Programs with Certaindex</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has unlocked their capabilities in advanced reasoning tasks like mathematical problem-solving, code generation, and legal analysis. Central to this progress are inference-time reasoning algorithms, which refine outputs by exploring multiple solution paths, at the cost of increasing compute demands and response latencies. Existing serving systems fail to adapt to the scaling behaviors of these algorithms or the varying difficulty of queries, leading to inefficient resource use and unmet latency targets. We present Dynasor, a system that optimizes inference-time compute for LLM reasoning queries. Unlike traditional engines, Dynasor tracks and schedules requests within reasoning queries and uses Certaindex, a proxy that measures statistical reasoning progress based on model certainty, to guide compute allocation dynamically. Dynasor co-adapts scheduling with reasoning progress: it allocates more compute to hard queries, reduces compute for simpler ones, and terminates unpromising queries early, balancing accuracy, latency, and cost. On diverse datasets and algorithms, Dynasor reduces compute by up to 50% in batch processing and sustaining 3.3x higher query rates or 4.7x tighter latency SLOs in online serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20942v1">Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 Presented at HI-AI@KDD, Human-Interpretable AI Workshop at the KDD 2024, 26th of August 2024, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      We propose an ontology-grounded approach to Knowledge Graph (KG) construction using Large Language Models (LLMs) on a knowledge base. An ontology is authored by generating Competency Questions (CQ) on knowledge base to discover knowledge scope, extracting relations from CQs, and attempt to replace equivalent relations by their counterpart in Wikidata. To ensure consistency and interpretability in the resulting KG, we ground generation of KG with the authored ontology based on extracted relations. Evaluation on benchmark datasets demonstrates competitive performance in knowledge graph construction task. Our work presents a promising direction for scalable KG construction pipeline with minimal human intervention, that yields high quality and human-interpretable KGs, which are interoperable with Wikidata semantics for potential knowledge base expansion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12075v2">WeatherDG: LLM-assisted Diffusion Model for Procedural Weather Generation in Domain-Generalized Semantic Segmentation</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel approach, namely WeatherDG, that can generate realistic, weather-diverse, and driving-screen images based on the cooperation of two foundation models, i.e, Stable Diffusion (SD) and Large Language Model (LLM). Specifically, we first fine-tune the SD with source data, aligning the content and layout of generated samples with real-world driving scenarios. Then, we propose a procedural prompt generation method based on LLM, which can enrich scenario descriptions and help SD automatically generate more diverse, detailed images. In addition, we introduce a balanced generation strategy, which encourages the SD to generate high-quality objects of tailed classes under various weather conditions, such as riders and motorcycles. This segmentation-model-agnostic method can improve the generalization ability of existing models by additionally adapting them with the generated synthetic data. Experiments on three challenging datasets show that our method can significantly improve the segmentation performance of different state-of-the-art models on target domains. Notably, in the setting of ''Cityscapes to ACDC'', our method improves the baseline HRDA by 13.9% in mIoU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20864v1">Enhancing Annotated Bibliography Generation with LLM Ensembles</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      This work proposes a novel approach to enhancing annotated bibliography generation through Large Language Model (LLM) ensembles. In particular, multiple LLMs in different roles -- controllable text generation, evaluation, and summarization -- are introduced and validated using a systematic methodology to enhance model performance in scholarly tasks. Output diversity among the ensemble that generates text is obtained using different LLM parameters, followed by an LLM acting as a judge to assess relevance, accuracy, and coherence. Responses selected by several combining strategies are then merged and refined through summarization and redundancy removal techniques. The preliminary experimental validation demonstrates that the combined outputs from the LLM ensemble improve coherence and relevance compared to individual responses, leading to a 38% improvement in annotation quality and a 51% reduction in content redundancy, thus highlighting the potential for automating complex scholarly tasks while maintaining high-quality standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20846v1">Are LLMs Really Not Knowledgable? Mining the Submerged Knowledge in LLMs' Memory</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise as potential knowledge bases, yet they often struggle with question-answering tasks and are prone to hallucinations. While previous research attributes these issues to knowledge gaps in the model's parameters, our investigation reveals a different phenomenon: LLMs often retain correct knowledge even when generating incorrect answers. Through analysis of model's internal representations, we find that correct answers frequently appear among high-probability tokens despite not being selected as final outputs. Based on this observation, we introduce Hits@k, a new metric to assess knowledge retention independent of expression accuracy. Our extensive experiments demonstrate that LLMs store significantly more knowledge than their QA performance suggests. Building on these findings, we develop SkipUnsure, a method to improve answer accuracy by leveraging detected but unexpressed knowledge. Experiments on both open-domain and specific-domain datasets show consistent improvements, with accuracy gains of up to 11.8% on DBPedia and 6.3% on IMDB, without requiring model retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18176v2">Molar: Multimodal LLMs with Collaborative Filtering Alignment for Enhanced Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      Sequential recommendation (SR) systems have evolved significantly over the past decade, transitioning from traditional collaborative filtering to deep learning approaches and, more recently, to large language models (LLMs). While the adoption of LLMs has driven substantial advancements, these models inherently lack collaborative filtering information, relying primarily on textual content data neglecting other modalities and thus failing to achieve optimal recommendation performance. To address this limitation, we propose Molar, a Multimodal large language sequential recommendation framework that integrates multiple content modalities with ID information to capture collaborative signals effectively. Molar employs an MLLM to generate unified item representations from both textual and non-textual data, facilitating comprehensive multimodal modeling and enriching item embeddings. Additionally, it incorporates collaborative filtering signals through a post-alignment mechanism, which aligns user representations from content-based and ID-based models, ensuring precise personalization and robust performance. By seamlessly combining multimodal content with collaborative filtering insights, Molar captures both user interests and contextual semantics, leading to superior recommendation accuracy. Extensive experiments validate that Molar significantly outperforms traditional and LLM-based baselines, highlighting its strength in utilizing multimodal data and collaborative signals for sequential recommendation tasks. The source code is available at https://anonymous.4open.science/r/Molar-8B06/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10424v2">LLM-as-an-Interviewer: Beyond Static Testing Through Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      We introduce LLM-as-an-Interviewer, a novel paradigm for evaluating large language models (LLMs). This approach leverages multi-turn interactions where the LLM interviewer actively provides feedback on responses and poses follow-up questions to the evaluated LLM. At the start of the interview, the LLM interviewer dynamically modifies datasets to generate initial questions, mitigating data contamination. We apply the LLM-as-an-Interviewer framework to evaluate six models on the MATH and DepthQA tasks. Our results show that the framework effectively provides insights into LLM performance, including the quality of initial responses, adaptability to feedback, and ability to address follow-up queries like clarification or additional knowledge requests. The framework also addresses key limitations of conventional methods like LLM-as-a-Judge, including verbosity bias and inconsistency across runs. Finally, we propose the Interview Report, which aggregates insights from the interview process, providing examples and a comprehensive analysis of the LLM's strengths and weaknesses. This report offers a detailed snapshot of the model's real-world applicability. The code for our framework is publicly available at https://github.com/interview-eval/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12543v3">LLM-based Translation Inference with Iterative Bilingual Understanding</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      The remarkable understanding and generation capabilities of large language models (LLMs) have greatly improved translation performance. However, incorrect understanding of the sentence to be translated can degrade translation quality. To address this issue, we proposed a novel Iterative Bilingual Understanding Translation (IBUT) method based on the cross-lingual capabilities of LLMs and the dual characteristics of translation tasks. The cross-lingual capability of LLMs enables the generation of contextual understanding for both the source and target languages separately. Furthermore, the dual characteristics allow IBUT to generate effective cross-lingual feedback, iteratively refining contextual understanding, thereby reducing errors and improving translation performance. Experimental results showed that the proposed IBUT outperforms several strong comparison methods, especially being generalized to multiple domains (e.g., news, commonsense, and cultural translation benchmarks).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03963v2">LLM-jp: A Cross-organizational Project for the Research and Development of Fully Open Japanese LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      This paper introduces LLM-jp, a cross-organizational project for the research and development of Japanese large language models (LLMs). LLM-jp aims to develop open-source and strong Japanese LLMs, and as of this writing, more than 1,500 participants from academia and industry are working together for this purpose. This paper presents the background of the establishment of LLM-jp, summaries of its activities, and technical reports on the LLMs developed by LLM-jp. For the latest activities, visit https://llm-jp.nii.ac.jp/en/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18819v2">LLM-assisted Vector Similarity Search</a></div>
    <div class="paper-meta">
      📅 2024-12-30
    </div>
    <details class="paper-abstract">
      As data retrieval demands become increasingly complex, traditional search methods often fall short in addressing nuanced and conceptual queries. Vector similarity search has emerged as a promising technique for finding semantically similar information efficiently. However, its effectiveness diminishes when handling intricate queries with contextual nuances. This paper explores a hybrid approach combining vector similarity search with Large Language Models (LLMs) to enhance search accuracy and relevance. The proposed two-step solution first employs vector similarity search to shortlist potential matches, followed by an LLM for context-aware ranking of the results. Experiments on structured datasets demonstrate that while vector similarity search alone performs well for straightforward queries, the LLM-assisted approach excels in processing complex queries involving constraints, negations, or conceptual requirements. By leveraging the natural language understanding capabilities of LLMs, this method improves the accuracy of search results for complex tasks without sacrificing efficiency. We also discuss real-world applications and propose directions for future research to refine and scale this technique for diverse datasets and use cases. Original article: https://engineering.grab.com/llm-assisted-vector-similarity-search
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14285v3">LLM-Personalize: Aligning LLM Planners with Human Preferences via Reinforced Self-Training for Housekeeping Robots</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 COLING 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown significant potential for robotics applications, particularly task planning, by harnessing their language comprehension and text generation capabilities. However, in applications such as household robotics, a critical gap remains in the personalization of these models to individual user preferences. We introduce LLM-Personalize, a novel framework with an optimization pipeline designed to personalize LLM planners for household robotics. Our LLM-Personalize framework features an LLM planner that performs iterative planning in multi-room, partially-observable household scenarios, making use of a scene graph constructed with local observations. The generated plan consists of a sequence of high-level actions which are subsequently executed by a controller. Central to our approach is the optimization pipeline, which combines imitation learning and iterative self-training to personalize the LLM planner. In particular, the imitation learning phase performs initial LLM alignment from demonstrations, and bootstraps the model to facilitate effective iterative self-training, which further explores and aligns the model to user preferences. We evaluate LLM-Personalize on Housekeep, a challenging simulated real-world 3D benchmark for household rearrangements, and show that LLM-Personalize achieves more than a 30 percent increase in success rate over existing LLM planners, showcasing significantly improved alignment with human preferences. Project page: https://gdg94.github.io/projectllmpersonalize/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20632v1">EVOLVE: Emotion and Visual Output Learning via LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-30
      | 💬 This work was presented at the WARN, Weighing the Benefits of Autonomous Robot Personalization, workshop at the 33rd IEEE RO-MAN 2024 conference
    </div>
    <details class="paper-abstract">
      Human acceptance of social robots is greatly effected by empathy and perceived understanding. This necessitates accurate and flexible responses to various input data from the user. While systems such as this can become increasingly complex as more states or response types are included, new research in the application of large language models towards human-robot interaction has allowed for more streamlined perception and reaction pipelines. LLM-selected actions and emotional expressions can help reinforce the realism of displayed empathy and allow for improved communication between the robot and user. Beyond portraying empathy in spoken or written responses, this shows the possibilities of using LLMs in actuated, real world scenarios. In this work we extend research in LLM-driven nonverbal behavior for social robots by considering more open-ended emotional response selection leveraging new advances in vision-language models, along with emotionally aligned motion and color pattern selections that strengthen conveyance of meaning and empathy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20613v1">Do Current Video LLMs Have Strong OCR Abilities? A Preliminary Study</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 Accepted by CoLing 2025 (The 31st International Conference on Computational Linguistics)
    </div>
    <details class="paper-abstract">
      With the rise of multimodal large language models, accurately extracting and understanding textual information from video content, referred to as video based optical character recognition (Video OCR), has become a crucial capability. This paper introduces a novel benchmark designed to evaluate the video OCR performance of multi-modal models in videos. Comprising 1,028 videos and 2,961 question-answer pairs, this benchmark proposes several key challenges through 6 distinct subtasks: (1) Recognition of text content itself and its basic visual attributes, (2)Semantic and Spatial Comprehension of OCR objects in videos (3) Dynamic Motion detection and Temporal Localization. We developed this benchmark using a semi-automated approach that integrates the OCR ability of image LLMs with manual refinement, balancing efficiency, cost, and data quality. Our resource aims to help advance research in video LLMs and underscores the need for improving OCR ability for video LLMs. The benchmark will be released on https://github.com/YuHuiGao/FG-Bench.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20595v1">Controlling Out-of-Domain Gaps in LLMs for Genre Classification and Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 The 31st International Conference on Computational Linguistics
    </div>
    <details class="paper-abstract">
      This study demonstrates that the modern generation of Large Language Models (LLMs, such as GPT-4) suffers from the same out-of-domain (OOD) performance gap observed in prior research on pre-trained Language Models (PLMs, such as BERT). We demonstrate this across two non-topical classification tasks: 1) genre classification and 2) generated text detection. Our results show that when demonstration examples for In-Context Learning (ICL) come from one domain (e.g., travel) and the system is tested on another domain (e.g., history), classification performance declines significantly. To address this, we introduce a method that controls which predictive indicators are used and which are excluded during classification. For the two tasks studied here, this ensures that topical features are omitted, while the model is guided to focus on stylistic rather than content-based attributes. This approach reduces the OOD gap by up to 20 percentage points in a few-shot setup. Straightforward Chain-of-Thought (CoT) methods, used as the baseline, prove insufficient, while our approach consistently enhances domain transfer performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18295v2">Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20505v1">Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 4 pages, 2 figures, accepted by The 1st Workshop on AI for Urban Planning (AAAI 2025's Workshop)
    </div>
    <details class="paper-abstract">
      Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20501v1">TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      Efficient parallelization of Large Language Models (LLMs) with long sequences is essential but challenging due to their significant computational and memory demands, particularly stemming from communication bottlenecks in attention mechanisms. While sequence parallelism (SP) has been introduced as a potential solution, existing methods often suffer from limited scalability or inefficiency, rendering their effectiveness. Ring-Attention demonstrates the potential for scaling sequence processing but faces significant limitations due to its reliance on peer-to-peer (P2P) communication and inefficient utilization of network resources. As the degree of SP increases, the quadratic decrease in computation time per step contrasts sharply with the linear reduction in communication volume, exacerbating communication bottlenecks. To address these challenges, we propose TokenRing, a fine-grained parallel framework that leverages bidirectional P2P communication to effectively overlap computation and data transmission. By partitioning the attention block and concurrently transmitting Query and block outputs (i.e., $block\_out$ and $block\_lse$) within a fully connected mesh topology, TokenRing achieves significant reductions in communication overhead and better load balancing. These innovations improve the scalability and efficiency of distributed Transformer models, particularly for long-context sequences. Experimental results demonstrate that TokenRing enhances throughput and reduces communication latency. Moreover, its design adapts seamlessly to various multi-GPU interconnect solutions, such as Huawei Ascend, ensuring broad compatibility and cost-effectiveness for distributed LLM inference and training. The code is available at: \url{https://github.com/ACA-Lab-SJTU/token-ring}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17881v2">AdaRankGrad: Adaptive Gradient-Rank and Moments for Memory-Efficient LLMs Training and Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      Training and fine-tuning large language models (LLMs) come with challenges related to memory and computational requirements due to the increasing size of the model weights and the optimizer states. Various techniques have been developed to tackle these challenges, such as low-rank adaptation (LoRA), which involves introducing a parallel trainable low-rank matrix to the fixed pre-trained weights at each layer. However, these methods often fall short compared to the full-rank weight training approach, as they restrict the parameter search to a low-rank subspace. This limitation can disrupt training dynamics and require a full-rank warm start to mitigate the impact. In this paper, we introduce a new method inspired by a phenomenon we formally prove: as training progresses, the rank of the estimated layer gradients gradually decreases, and asymptotically approaches rank one. Leveraging this, our approach involves adaptively reducing the rank of the gradients during Adam optimization steps, using an efficient online-updating low-rank projections rule. We further present a randomized SVD scheme for efficiently finding the projection matrix. Our technique enables full-parameter fine-tuning with adaptive low-rank gradient updates, significantly reducing overall memory requirements during training compared to state-of-the-art methods while improving model performance in both pretraining and fine-tuning. Finally, we provide a convergence analysis of our method and demonstrate its merits for training and fine-tuning language and biological foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20440v1">Enhancing Entertainment Translation for Indian Languages using Adaptive Context, Style and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 Accepted to AAAI'25
    </div>
    <details class="paper-abstract">
      We address the challenging task of neural machine translation (NMT) in the entertainment domain, where the objective is to automatically translate a given dialogue from a source language content to a target language. This task has various applications, particularly in automatic dubbing, subtitling, and other content localization tasks, enabling source content to reach a wider audience. Traditional NMT systems typically translate individual sentences in isolation, without facilitating knowledge transfer of crucial elements such as the context and style from previously encountered sentences. In this work, we emphasize the significance of these fundamental aspects in producing pertinent and captivating translations. We demonstrate their significance through several examples and propose a novel framework for entertainment translation, which, to our knowledge, is the first of its kind. Furthermore, we introduce an algorithm to estimate the context and style of the current session and use these estimations to generate a prompt that guides a Large Language Model (LLM) to generate high-quality translations. Our method is both language and LLM-agnostic, making it a general-purpose tool. We demonstrate the effectiveness of our algorithm through various numerical studies and observe significant improvement in the COMET scores over various state-of-the-art LLMs. Moreover, our proposed method consistently outperforms baseline LLMs in terms of win-ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20414v1">Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 6 pages, 1 table, AICCONF '24: Cognitive Models and Artificial Intelligence Conference, Istanbul, Turkey
    </div>
    <details class="paper-abstract">
      The integration of advanced Natural Language Processing (NLP) methodologies and Large Language Models (LLMs) has significantly enhanced the extraction and analysis of geospatial data from multilingual texts, impacting sectors such as national and international security. This paper presents a comprehensive evaluation of leading NLP models -- SpaCy, XLM-RoBERTa, mLUKE, GeoLM -- and LLMs, specifically OpenAI's GPT 3.5 and GPT 4, within the context of multilingual geo-entity detection. Utilizing datasets from Telegram channels in English, Russian, and Arabic, we examine the performance of these models through metrics such as accuracy, precision, recall, and F1 scores, to assess their effectiveness in accurately identifying geospatial references. The analysis exposes each model's distinct advantages and challenges, underscoring the complexities involved in achieving precise geo-entity identification across varied linguistic landscapes. The conclusions drawn from this experiment aim to direct the enhancement and creation of more advanced and inclusive NLP tools, thus advancing the field of geospatial analysis and its application to global security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01085v3">Explaining Length Bias in LLM-Based Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as judges, particularly in preference comparisons, has become widespread, but this reveals a notable bias towards longer responses, undermining the reliability of such evaluations. To better understand such bias, we propose to decompose the preference evaluation metric, specifically the win rate, into two key components: desirability and information mass, where the former is length-independent and related to trustworthiness such as correctness, toxicity, and consistency, and the latter is length-dependent and represents the amount of information in the response. We empirically demonstrated the decomposition through controlled experiments and found that response length impacts evaluations by influencing information mass. To derive a reliable evaluation metric that assesses content quality without being confounded by response length, we propose AdapAlpaca, a simple yet effective adjustment to win rate measurement. Specifically, AdapAlpaca ensures a fair comparison of response quality by aligning the lengths of reference and test model responses under equivalent length intervals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10064v4">Revolutionizing Bridge Operation and Maintenance with LLM-based Agents: An Overview of Applications and Insights</a></div>
    <div class="paper-meta">
      📅 2024-12-29
    </div>
    <details class="paper-abstract">
      In various industrial fields of human social development, people have been exploring methods aimed at freeing human labor. Constructing LLM-based agents is considered to be one of the most effective tools to achieve this goal. Agent, as a kind of human-like intelligent entity with the ability of perception, planning, decision-making, and action, has created great production value in many fields. However, the bridge O&M field shows a relatively low level of intelligence compared to other industries. Nevertheless, the bridge O&M field has developed numerous intelligent inspection devices, machine learning algorithms, and autonomous evaluation and decision-making methods, which provide a feasible basis for breakthroughs in artificial intelligence in this field. The aim of this study is to explore the impact of AI bodies based on large-scale language models on the field of bridge O&M and to analyze the potential challenges and opportunities it brings to the core tasks of bridge O&M. Through in-depth research and analysis, this paper expects to provide a more comprehensive perspective for understanding the application of intelligentsia in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20331v1">Mind the Data Gap: Bridging LLMs to Enterprise Data Integration</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 CIDR'25
    </div>
    <details class="paper-abstract">
      Leading large language models (LLMs) are trained on public data. However, most of the world's data is dark data that is not publicly accessible, mainly in the form of private organizational or enterprise data. We show that the performance of methods based on LLMs seriously degrades when tested on real-world enterprise datasets. Current benchmarks, based on public data, overestimate the performance of LLMs. We release a new benchmark dataset, the GOBY Benchmark, to advance discovery in enterprise data integration. Based on our experience with this enterprise benchmark, we propose techniques to uplift the performance of LLMs on enterprise data, including (1) hierarchical annotation, (2) runtime class-learning, and (3) ontology synthesis. We show that, once these techniques are deployed, the performance on enterprise data becomes on par with that of public data. The Goby benchmark can be obtained at https://goby-benchmark.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08854v2">Hybrid LLM-DDQN based Joint Optimization of V2I Communication and Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-12-29
      | 💬 Submission for possible publication
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have received considerable interest recently due to their outstanding reasoning and comprehension capabilities. This work explores applying LLMs to vehicular networks, aiming to jointly optimize vehicle-to-infrastructure (V2I) communications and autonomous driving (AD) policies. We deploy LLMs for AD decision-making to maximize traffic flow and avoid collisions for road safety, and a double deep Q-learning algorithm (DDQN) is used for V2I optimization to maximize the received data rate and reduce frequent handovers. In particular, for LLM-enabled AD, we employ the Euclidean distance to identify previously explored AD experiences, and then LLMs can learn from past good and bad decisions for further improvement. Then, LLM-based AD decisions will become part of states in V2I problems, and DDQN will optimize the V2I decisions accordingly. After that, the AD and V2I decisions are iteratively optimized until convergence. Such an iterative optimization approach can better explore the interactions between LLMs and conventional reinforcement learning techniques, revealing the potential of using LLMs for network optimization and management. Finally, the simulations demonstrate that our proposed hybrid LLM-DDQN approach outperforms the conventional DDQN algorithm, showing faster convergence and higher average rewards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20297v1">FaGeL: Fabric LLMs Agent empowered Embodied Intelligence Evolution with Autonomous Human-Machine Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have enhanced the reasoning capabilities of embodied agents, driving progress toward AGI-powered robotics. While LLMs have been applied to tasks like semantic reasoning and task generalization, their potential in open physical space exploration remains underexplored. This paper introduces FaGeL (Fabric aGent empowered by embodied intelligence with LLMs), an embodied agent integrating smart fabric technology for seamless, non-intrusive human-agent interaction. FaGeL autonomously generates tasks using multimodal data from wearable and ambient sensors, refining its behavior based on implicit human feedback in generated text, without explicit ratings or preferences. We also introduce a token-level saliency map to visualize LLM fine-tuning, enhancing the interpretability of token-level alignment. The system leverages dual feedback mechanisms to improve token-level alignment and addresses challenges in non-intrusive human-machine interaction and cognition evolution. Our contributions include FaGeL's development, the DualCUT algorithm for AI alignment, and experimental validation in cooperative tasks, demonstrating FaGeL's ability to adapt and evolve autonomously through implicit feedback. In the future, we plan to explore FaGeL's scalability in dynamic environments and its integration with other AI systems to develop AGI agents that adapt seamlessly to diverse human needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06205v1">Leveraging Edge Intelligence and LLMs to Advance 6G-Enabled Internet of Automated Defense Vehicles</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 8 pages, 5 figures, under (secondary/revised) review in IEEE Internet of Things Magazine
    </div>
    <details class="paper-abstract">
      The evolution of Artificial Intelligence (AI) and its subset Deep Learning (DL), has profoundly impacted numerous domains, including autonomous driving. The integration of autonomous driving in military settings reduces human casualties and enables precise and safe execution of missions in hazardous environments while allowing for reliable logistics support without the risks associated with fatigue-related errors. However, relying on autonomous driving solely requires an advanced decision-making model that is adaptable and optimum in any situation. Considering the presence of numerous interconnected autonomous vehicles in mission-critical scenarios, Ultra-Reliable Low Latency Communication (URLLC) is vital for ensuring seamless coordination, real-time data exchange, and instantaneous response to dynamic driving environments. The advent of 6G strengthens the Internet of Automated Defense Vehicles (IoADV) concept within the realm of Internet of Military Defense Things (IoMDT) by enabling robust connectivity, crucial for real-time data exchange, advanced navigation, and enhanced safety features through IoADV interactions. On the other hand, a critical advancement in this space is using pre-trained Generative Large Language Models (LLMs) for decision-making and communication optimization for autonomous driving. Hence, this work presents opportunities and challenges with a vision of realizing the full potential of these technologies in critical defense applications, especially through the advancement of IoADV and its role in enhancing autonomous military operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20251v1">ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      The rapid development of LLMs has sparked extensive research into their factual knowledge. Current works claim that LLMs fall short on questions requiring less frequent knowledge. However, their proof is incomplete since they only study the influence of entity frequency, which can not fully represent knowledge frequency. So we introduce ComparisonQA benchmark, containing 283K abstract questions, each instantiated by a pair of high-frequency and low-frequency entities. It ensures a controllable comparison because the difference of knowledge frequency between such a pair is only related to entity frequency. In addition, to avoid possible semantic shortcuts, which is a severe problem of current LLMs study, we design a two-round method for knowledge robustness measurement utilizing both correctness and uncertainty. Experiments reveal that LLMs exhibit particularly low robustness regarding low-frequency knowledge, and GPT-4o is even the worst under this measurement. Besides, we introduce an automatic method to filter out questions with low-quality and shortcuts to form ComparisonQA-Hard. We find that uncertainty effectively identifies such questions while maintaining the data size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20227v1">LLM Reasoning Engine: Specialized Training for Enhanced Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance in various natural language processing tasks but face challenges in mathematical reasoning, where complex problem-solving requires both linguistic understanding and mathematical reasoning skills. Existing approaches to address this challenge often rely on ensemble methods and suffer from the problem of data scarcity in target domains. In this work, we present a novel method to enhance LLMs' capabilities in mathematical reasoning tasks. Motivated by the need to bridge this gap, our approach incorporates a question paraphrase strategy, which aims at diversifying the linguistic forms of mathematical questions to improve generalization. Additionally, specialized training objectives are employed to guide the model's learning process, focusing on enhancing its understanding of mathematical concepts and reasoning processes. We conduct experiments on four datasets using different LLMs, and demonstrate the effectiveness of our approach in improving LLMs' performance on mathematical reasoning tasks. Our findings underscore the significance of our methodology in the advancement of large language models and its potential implications for real-world applications that require mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06294v3">ArgMed-Agents: Explainable Clinical Decision Reasoning with LLM Disscusion via Argumentation Schemes</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      There are two main barriers to using large language models (LLMs) in clinical reasoning. Firstly, while LLMs exhibit significant promise in Natural Language Processing (NLP) tasks, their performance in complex reasoning and planning falls short of expectations. Secondly, LLMs use uninterpretable methods to make clinical decisions that are fundamentally different from the clinician's cognitive processes. This leads to user distrust. In this paper, we present a multi-agent framework called ArgMed-Agents, which aims to enable LLM-based agents to make explainable clinical decision reasoning through interaction. ArgMed-Agents performs self-argumentation iterations via Argumentation Scheme for Clinical Discussion (a reasoning mechanism for modeling cognitive processes in clinical reasoning), and then constructs the argumentation process as a directed graph representing conflicting relationships. Ultimately, use symbolic solver to identify a series of rational and coherent arguments to support decision. We construct a formal model of ArgMed-Agents and present conjectures for theoretical guarantees. ArgMed-Agents enables LLMs to mimic the process of clinical argumentative reasoning by generating explanations of reasoning in a self-directed manner. The setup experiments show that ArgMed-Agents not only improves accuracy in complex clinical decision reasoning problems compared to other prompt methods, but more importantly, it provides users with decision explanations that increase their confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20185v1">Pushing the Envelope of Low-Bit LLM via Dynamic Error Compensation</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      Quantization of Large Language Models (LLMs) has recently gained popularity, particularly for on-device settings with limited hardware resources. While efficient, quantization inevitably degrades model quality, especially in aggressive low-bit settings such as 3-bit and 4-bit precision. In this paper, we propose QDEC, an inference scheme that improves the quality of low-bit LLMs while preserving the key benefits of quantization: GPU memory savings and inference latency reduction. QDEC stores the residual matrix -- the difference between full-precision and quantized weights -- in CPU, and dynamically fetches the residuals for only a small portion of the weights. This portion corresponds to the salient channels, marked by activation outliers, with the fetched residuals helping to correct quantization errors in these channels. Salient channels are identified dynamically at each decoding step by analyzing the input activations -- this allows for the adaptation to the dynamic nature of activation distribution, and thus maximizes the effectiveness of error compensation. We demonstrate the effectiveness of QDEC by augmenting state-of-the-art quantization methods. For example, QDEC reduces the perplexity of a 3-bit Llama-3-8B-Instruct model from 10.15 to 9.12 -- outperforming its 3.5-bit counterpart -- while adding less than 0.0003\% to GPU memory usage and incurring only a 1.7\% inference slowdown on NVIDIA RTX 4050 Mobile GPU. The code will be publicly available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14033v2">LLM-based Multi-Agent Systems: Techniques and Business Perspectives</a></div>
    <div class="paper-meta">
      📅 2024-12-28
    </div>
    <details class="paper-abstract">
      In the era of (multi-modal) large language models, most operational processes can be reformulated and reproduced using LLM agents. The LLM agents can perceive, control, and get feedback from the environment so as to accomplish the given tasks in an autonomous manner. Besides the environment-interaction property, the LLM agents can call various external tools to ease the task completion process. The tools can be regarded as a predefined operational process with private or real-time knowledge that does not exist in the parameters of LLMs. As a natural trend of development, the tools for calling are becoming autonomous agents, thus the full intelligent system turns out to be a LLM-based Multi-Agent System (LaMAS). Compared to the previous single-LLM-agent system, LaMAS has the advantages of i) dynamic task decomposition and organic specialization, ii) higher flexibility for system changing, iii) proprietary data preserving for each participating entity, and iv) feasibility of monetization for each entity. This paper discusses the technical and business landscapes of LaMAS. To support the ecosystem of LaMAS, we provide a preliminary version of such LaMAS protocol considering technical requirements, data privacy, and business incentives. As such, LaMAS would be a practical solution to achieve artificial collective intelligence in the near future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03350v2">A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 78 pages, 32 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated emergent abilities in text generation, question answering, and reasoning, facilitating various tasks and domains. Despite their proficiency in various tasks, LLMs like PaLM 540B and Llama-3.1 405B face limitations due to large parameter sizes and computational demands, often requiring cloud API use which raises privacy concerns, limits real-time applications on edge devices, and increases fine-tuning costs. Additionally, LLMs often underperform in specialized domains such as healthcare and law due to insufficient domain-specific knowledge, necessitating specialized models. Therefore, Small Language Models (SLMs) are increasingly favored for their low inference latency, cost-effectiveness, efficient development, and easy customization and adaptability. These models are particularly well-suited for resource-limited environments and domain knowledge acquisition, addressing LLMs' challenges and proving ideal for applications that require localized data handling for privacy, minimal inference latency for efficiency, and domain knowledge acquisition through lightweight fine-tuning. The rising demand for SLMs has spurred extensive research and development. However, a comprehensive survey investigating issues related to the definition, acquisition, application, enhancement, and reliability of SLM remains lacking, prompting us to conduct a detailed survey on these topics. The definition of SLMs varies widely, thus to standardize, we propose defining SLMs by their capability to perform specialized tasks and suitability for resource-constrained settings, setting boundaries based on the minimal size for emergent abilities and the maximum size sustainable under resource constraints. For other aspects, we provide a taxonomy of relevant models/methods and develop general frameworks for each category to enhance and utilize SLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20087v1">On the Validity of Traditional Vulnerability Scoring Systems for Adversarial Attacks against LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 101 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This research investigates the effectiveness of established vulnerability metrics, such as the Common Vulnerability Scoring System (CVSS), in evaluating attacks against Large Language Models (LLMs), with a focus on Adversarial Attacks (AAs). The study explores the influence of both general and specific metric factors in determining vulnerability scores, providing new perspectives on potential enhancements to these metrics. This study adopts a quantitative approach, calculating and comparing the coefficient of variation of vulnerability scores across 56 adversarial attacks on LLMs. The attacks, sourced from various research papers, and obtained through online databases, were evaluated using multiple vulnerability metrics. Scores were determined by averaging the values assessed by three distinct LLMs. The results indicate that existing scoring-systems yield vulnerability scores with minimal variation across different attacks, suggesting that many of the metric factors are inadequate for assessing adversarial attacks on LLMs. This is particularly true for context-specific factors or those with predefined value sets, such as those in CVSS. These findings support the hypothesis that current vulnerability metrics, especially those with rigid values, are limited in evaluating AAs on LLMs, highlighting the need for the development of more flexible, generalized metrics tailored to such attacks. This research offers a fresh analysis of the effectiveness and applicability of established vulnerability metrics, particularly in the context of Adversarial Attacks on Large Language Models, both of which have gained significant attention in recent years. Through extensive testing and calculations, the study underscores the limitations of these metrics and opens up new avenues for improving and refining vulnerability assessment frameworks specifically tailored for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10835v5">Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 Accepted by SIGKDD Explorations Newsletter
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been applied in many fields and have developed rapidly in recent years. As a classic machine learning task, time series forecasting has recently been boosted by LLMs. Recent works treat large language models as \emph{zero-shot} time series reasoners without further fine-tuning, which achieves remarkable performance. However, there are some unexplored research problems when applying LLMs for time series forecasting under the zero-shot setting. For instance, the LLMs' preferences for the input time series are less understood. In this paper, by comparing LLMs with traditional time series forecasting models, we observe many interesting properties of LLMs in the context of time series forecasting. First, our study shows that LLMs perform well in predicting time series with clear patterns and trends, but face challenges with datasets lacking periodicity. This observation can be explained by the ability of LLMs to recognize the underlying period within datasets, which is supported by our experiments. In addition, the input strategy is investigated, and it is found that incorporating external knowledge and adopting natural language paraphrases substantially improve the predictive performance of LLMs for time series. Overall, our study contributes insight into LLMs' advantages and limitations in time series forecasting under different conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00847v3">The Design of an LLM-powered Unstructured Analytics System</a></div>
    <div class="paper-meta">
      📅 2024-12-28
      | 💬 Included in the proceedings of The Conference on Innovative Data Systems Research (CIDR) 2025
    </div>
    <details class="paper-abstract">
      LLMs demonstrate an uncanny ability to process unstructured data, and as such, have the potential to go beyond search and run complex, semantic analyses at scale. We describe the design of an unstructured analytics system, Aryn, and the tenets and use cases that motivate its design. With Aryn, users specify queries in natural language and the system automatically determines a semantic plan and executes it to compute an answer from a large collection of unstructured documents. At the core of Aryn is Sycamore, a declarative document processing engine, that provides a reliable distributed abstraction called DocSets. Sycamore allows users to analyze, enrich, and transform complex documents at scale. Aryn includes Luna, a query planner that translates natural language queries to Sycamore scripts, and DocParse, which takes raw PDFs and document images, and converts them to DocSets for downstream processing. We show how these pieces come together to achieve better accuracy than RAG on analytics queries over real world reports from the National Transportation Safety Board (NTSB). Also, given current limitations of LLMs, we argue that an analytics system must provide explainability to be practical, and show how Aryn's user interface does this to help build trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19926v1">Right vs. Right: Can LLMs Make Tough Choices?</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      An ethical dilemma describes a choice between two "right" options involving conflicting moral values. We present a comprehensive evaluation of how LLMs navigate ethical dilemmas. Specifically, we investigate LLMs on their (1) sensitivity in comprehending ethical dilemmas, (2) consistency in moral value choice, (3) consideration of consequences, and (4) ability to align their responses to a moral value preference explicitly or implicitly specified in a prompt. Drawing inspiration from a leading ethical framework, we construct a dataset comprising 1,730 ethical dilemmas involving four pairs of conflicting values. We evaluate 20 well-known LLMs from six families. Our experiments reveal that: (1) LLMs exhibit pronounced preferences between major value pairs, and prioritize truth over loyalty, community over individual, and long-term over short-term considerations. (2) The larger LLMs tend to support a deontological perspective, maintaining their choices of actions even when negative consequences are specified. (3) Explicit guidelines are more effective in guiding LLMs' moral choice than in-context examples. Lastly, our experiments highlight the limitation of LLMs in comprehending different formulations of ethical dilemmas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19906v1">Evaluate Summarization in Fine-Granularity: Auto Evaluation with LLM</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Due to the exponential growth of information and the need for efficient information consumption the task of summarization has gained paramount importance. Evaluating summarization accurately and objectively presents significant challenges, particularly when dealing with long and unstructured texts rich in content. Existing methods, such as ROUGE (Lin, 2004) and embedding similarities, often yield scores that have low correlation with human judgements and are also not intuitively understandable, making it difficult to gauge the true quality of the summaries. LLMs can mimic human in giving subjective reviews but subjective scores are hard to interpret and justify. They can be easily manipulated by altering the models and the tones of the prompts. In this paper, we introduce a novel evaluation methodology and tooling designed to address these challenges, providing a more comprehensive, accurate and interpretable assessment of summarization outputs. Our method (SumAutoEval) proposes and evaluates metrics at varying granularity levels, giving objective scores on 4 key dimensions such as completeness, correctness, Alignment and readability. We empirically demonstrate, that SumAutoEval enhances the understanding of output quality with better human correlation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19770v1">Fortran2CPP: Automating Fortran-to-C++ Migration using LLMs via Multi-Turn Dialogue and Dual-Agent Integration</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Migrating Fortran code to C++ is a common task for many scientific computing teams, driven by the need to leverage modern programming paradigms, enhance cross-platform compatibility, and improve maintainability. Automating this translation process using large language models (LLMs) has shown promise, but the lack of high-quality, specialized datasets has hindered their effectiveness. In this paper, we address this challenge by introducing a novel multi-turn dialogue dataset, Fortran2CPP, specifically designed for Fortran-to-C++ code migration. Our dataset, significantly larger than existing alternatives, is generated using a unique LLM-driven, dual-agent pipeline incorporating iterative compilation, execution, and code repair to ensure high quality and functional correctness. To demonstrate the effectiveness of our dataset, we fine-tuned several open-weight LLMs on Fortran2CPP and evaluated their performance on two independent benchmarks. Fine-tuning on our dataset led to remarkable gains, with models achieving up to a 3.31x increase in CodeBLEU score and a 92\% improvement in compilation success rate. This highlights the dataset's ability to enhance both the syntactic accuracy and compilability of the translated C++ code. Our dataset and model have been open-sourced and are available on our public GitHub repository\footnote{\url{https://github.com/HPC-Fortran2CPP/Fortran2Cpp}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01366v2">CHESS: Optimizing LLM Inference via Channel-Wise Thresholding and Selective Sparsification</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on edge devices presents significant challenges due to the substantial computational overhead and memory requirements. Activation sparsification can mitigate these resource challenges by reducing the number of activated neurons during inference. Existing methods typically employ thresholding-based sparsification based on the statistics of activation tensors. However, they do not model the impact of activation sparsification on performance, resulting in suboptimal performance degradation. To address the limitations, this paper reformulates the activation sparsification problem to explicitly capture the relationship between activation sparsity and model performance. Then, this paper proposes CHESS, a general activation sparsification approach via CHannel-wise thrEsholding and Selective Sparsification. First, channel-wise thresholding assigns a unique threshold to each activation channel in the feed-forward network (FFN) layers. Then, selective sparsification involves applying thresholding-based activation sparsification to specific layers within the attention modules. Finally, we detail the implementation of sparse kernels to accelerate LLM inference. Experimental results demonstrate that the proposed CHESS achieves lower performance degradation over eight downstream tasks while activating fewer parameters than existing methods, thus speeding up the LLM inference by up to 1.27x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02946v5">Data Poisoning in LLMs: Jailbreak-Tuning and Scaling Laws</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      LLMs produce harmful and undesirable behavior when trained on poisoned datasets that contain a small fraction of corrupted or harmful data. We develop a new attack paradigm, jailbreak-tuning, that combines data poisoning with jailbreaking to fully bypass state-of-the-art safeguards and make models like GPT-4o comply with nearly any harmful request. Our experiments suggest this attack represents a paradigm shift in vulnerability elicitation, producing differences in refusal rates as much as 60+ percentage points compared to normal fine-tuning. Given this demonstration of how data poisoning vulnerabilities persist and can be amplified, we investigate whether these risks will likely increase as models scale. We evaluate three threat models - malicious fine-tuning, imperfect data curation, and intentional data contamination - across 24 frontier LLMs ranging from 1.5 to 72 billion parameters. Our experiments reveal that larger LLMs are significantly more susceptible to data poisoning, learning harmful behaviors from even minimal exposure to harmful data more quickly than smaller models. These findings underscore the need for leading AI companies to thoroughly red team fine-tuning APIs before public release and to develop more robust safeguards against data poisoning, particularly as models continue to scale in size and capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13362v3">Lusifer: LLM-based User SImulated Feedback Environment for online Recommender systems</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Training reinforcement learning-based recommender systems is often hindered by the lack of dynamic and realistic user interactions. To address this limitation, we introduce Lusifer, a novel environment leveraging Large Language Models (LLMs) to generate simulated user feedback. Lusifer synthesizes user profiles and interaction histories to simulate responses and behaviors toward recommended items, with profiles updated after each rating to reflect evolving user characteristics. Utilizing the MovieLens dataset as a proof of concept, we limited our implementation to the last 40 interactions for each user, representing approximately 39% and 22% of the training sets, to focus on recent user behavior. For consistency and to gain insights into the performance of traditional methods with limited data, we implemented baseline approaches using the same data subset. Our results demonstrate that Lusifer accurately emulates user behavior and preferences, even with reduced training data having an RMSE of 1.3 across various test sets. This paper presents Lusifer's operational pipeline, including prompt generation and iterative user profile updates, and compares its performance against baseline methods. The findings validate Lusifer's ability to produce realistic dynamic feedback and suggest that it offers a scalable and adjustable framework for user simulation in online reinforcement learning recommender systems for future studies, particularly when training data is limited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01880v1">Long Context vs. RAG for LLMs: An Evaluation and Revisits</a></div>
    <div class="paper-meta">
      📅 2024-12-27
      | 💬 14 pages excluding references and appendix
    </div>
    <details class="paper-abstract">
      Extending context windows (i.e., Long Context, LC) and using retrievers to selectively access relevant information (i.e., Retrieval-Augmented Generation, RAG) are the two main strategies to enable LLMs to incorporate extremely long external contexts. This paper revisits recent studies on this topic, highlighting their key insights and discrepancies. We then provide a more comprehensive evaluation by filtering out questions answerable without external context, identifying the most effective retrieval methods, and expanding the datasets. We show that LC generally outperforms RAG in question-answering benchmarks, especially for Wikipedia-based questions. Summarization-based retrieval performs comparably to LC, while chunk-based retrieval lags behind. However, RAG has advantages in dialogue-based and general question queries. These insights underscore the trade-offs between RAG and LC strategies, offering guidance for future optimization of LLMs with external knowledge sources. We also provide an in-depth discussion on this topic, highlighting the overlooked importance of context relevance in existing studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11843v3">Preemptive Detection and Correction of Misaligned Actions in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Deploying LLM-based agents in real-life applications often faces a critical challenge: the misalignment between agents' behavior and user intent. Such misalignment may lead agents to unintentionally execute critical actions that carry negative outcomes (e.g., accidentally triggering a "buy-now" in web shopping), resulting in undesirable or even irreversible consequences. Although addressing these issues is crucial, the preemptive detection and correction of misaligned actions remains relatively underexplored. To fill this gap, we introduce InferAct, a novel approach that leverages the belief reasoning ability of LLMs, grounded in Theory-of-Mind, to detect misaligned actions before execution. Once the misalignment is detected, InferAct alerts users for timely correction, preventing adverse outcomes and enhancing the reliability of LLM agents' decision-making processes. Experiments on three widely used tasks demonstrate that InferAct achieves up to 20% improvements on Marco-F1 against baselines in misaligned action detection. An in-depth evaluation of misalignment correction further highlights InferAct's effectiveness in improving agent alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19493v3">Official-NV: An LLM-Generated News Video Dataset for Multimodal Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      News media, especially video news media, have penetrated into every aspect of daily life, which also brings the risk of fake news. Therefore, multimodal fake news detection has recently garnered increased attention. However, the existing datasets are comprised of user-uploaded videos and contain an excess amounts of superfluous data, which introduces noise into the model training process. To address this issue, we construct a dataset named Official-NV, comprising officially published news videos. The crawl officially published videos are augmented through the use of LLMs-based generation and manual verification, thereby expanding the dataset. We also propose a new baseline model called OFNVD, which captures key information from multimodal features through a GLU attention mechanism and performs feature enhancement and modal aggregation via a cross-modal Transformer. Benchmarking the dataset and baselines demonstrates the effectiveness of our model in multimodal news detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11843v2">From Commands to Prompts: LLM-based Semantic File System for AIOS</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in the development of intelligent applications and systems such as LLM-based agents and agent operating systems (AIOS). However, when these applications and systems interact with the underlying file system, the file system still remains the traditional paradigm: reliant on manual navigation through precise commands. This paradigm poses a bottleneck to the usability of these systems as users are required to navigate complex folder hierarchies and remember cryptic file names. To address this limitation, we propose an LLM-based semantic file system ( LSFS ) for prompt-driven file management. Unlike conventional approaches, LSFS incorporates LLMs to enable users or agents to interact with files through natural language prompts, facilitating semantic file management. At the macro-level, we develop a comprehensive API set to achieve semantic file management functionalities, such as semantic file retrieval, file update monitoring and summarization, and semantic file rollback). At the micro-level, we store files by constructing semantic indexes for them, design and implement syscalls of different semantic operations (e.g., CRUD, group by, join) powered by vector database. Our experiments show that LSFS offers significant improvements over traditional file systems in terms of user convenience, the diversity of supported functions, and the accuracy and efficiency of file operations. Additionally, with the integration of LLM, our system enables more intelligent file management tasks, such as content summarization and version comparison, further enhancing its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19513v1">Confidence v.s. Critique: A Decomposition of Self-Correction Capability for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-27
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can correct their self-generated responses, but a decline in accuracy after self-correction is also witnessed. To have a deeper understanding of self-correction, we endeavor to decompose, evaluate, and analyze the self-correction behaviors of LLMs. By enumerating and analyzing answer correctness before and after self-correction, we decompose the self-correction capability into confidence (being confident to correct answers) and critique (turning wrong answers to correct) capabilities, and propose two metrics from a probabilistic perspective to measure these 2 capabilities, along with another metric for overall self-correction capability evaluation. Based on our decomposition and evaluation metrics, we conduct extensive experiments and draw some empirical conclusions. For example, we find different models can exhibit distinct behaviors: some models are confident while others are more critical. We also find the trade-off between the two capabilities (i.e. improving one can lead to a decline in the other) when manipulating model self-correction behavior by prompts or in-context learning. Further, we find a simple yet efficient strategy to improve self-correction capability by transforming Supervision Fine-Tuning (SFT) data format, and our strategy outperforms vanilla SFT in both capabilities and achieves much higher accuracy after self-correction. Our code will be publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.19512v1">Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) for downstream tasks is a widely adopted approach, but it often leads to safety degradation in safety-aligned LLMs. Currently, many solutions address this issue by incorporating additional safety data, which can be impractical in many cases. In this paper, we address the question: How can we improve downstream task performance while preserving safety in LLMs without relying on additional safety data? We propose a simple and effective method that maintains the inherent safety of LLMs while enhancing their downstream task performance: merging the weights of pre- and post-fine-tuned safety-aligned models. Experimental results across various downstream tasks, models, and merging methods demonstrate that this approach effectively mitigates safety degradation while improving downstream task performance, offering a practical solution for adapting safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17378v3">A Text is Worth Several Tokens: Text Embedding from LLMs Secretly Aligns Well with The Key Tokens</a></div>
    <div class="paper-meta">
      📅 2024-12-27
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Text embeddings from large language models (LLMs) have achieved excellent results in tasks such as information retrieval, semantic textual similarity, etc. In this work, we show an interesting finding: when feeding a text into the LLM-based embedder, the obtained text embedding will be able to be aligned with the key tokens in the input text. We first fully analyze this phenomenon on eight LLM-based embedders and show that this phenomenon is universal and is not affected by model architecture, training strategy, and embedding method. With a deeper analysis, we find that the main change in embedding space between these embedders and their LLM backbones is in the first principal component. By adjusting the first principal component, we can align text embedding with the key tokens. Finally, we give several examples to demonstrate the vast application potential of this finding: (1) we propose a simple and practical sparse retrieval method based on the aligned tokens, which can achieve 80% of the dense retrieval effect of the same model while reducing the computation significantly; (2) we show that our findings provide a novel perspective to help understand novel technologies (e.g., instruction-following embedding) and fuzzy concepts (e.g., semantic relatedness vs. similarity) in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09032v3">CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences</a></div>
    <div class="paper-meta">
      📅 2024-12-27
    </div>
    <details class="paper-abstract">
      Evaluating the alignment of large language models (LLMs) with user-defined coding preferences is a challenging endeavour that requires a deep assessment of LLMs' outputs. Existing methods and benchmarks rely primarily on automated metrics and static analysis tools, which often fail to capture the nuances of user instructions and LLM outputs. To address this gap, we propose using the LLM-as-a-Judge methodology to evaluate the alignment of LLMs with coding preferences. Based on this approach, we present CodeUltraFeedback, a comprehensive dataset designed to facilitate the evaluation and improvement of LLM alignment. CodeUltraFeedback consists of 10,000 coding instructions, each annotated with four responses generated from a diverse pool of 14 LLMs. These responses are ranked based on five distinct coding preferences using GPT-3.5 as a judge, providing both numerical scores and detailed textual feedback. Our analysis of CodeUltraFeedback reveals that responses from GPT-3.5 and GPT-4 are generally preferred over those from open-weight LLMs, highlighting significant differences in alignment between closed and open-weight models. In turn, we explore the usage of CodeUltraFeedback as feedback data to fine-tune and align CodeLlama-7B-Instruct using supervised fine-tuning (SFT) and reinforcement learning from AI feedback (RLAIF) with direct preference optimization (DPO). The resulting aligned CodeLlama-7B-Instruct model outperforms larger LLMs in terms of alignment with coding preferences and shows improved functional correctness on the HumanEval+ benchmark compared to the original instruct model. Therefore, our contributions bridge the gap in preference tuning of LLMs for code and set the stage for further advancements in model alignment and RLAIF in automated software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13168v4">LLMs for Knowledge Graph Construction and Reasoning: Recent Capabilities and Future Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-12-26
      | 💬 World Wide Web Journal
    </div>
    <details class="paper-abstract">
      This paper presents an exhaustive quantitative and qualitative evaluation of Large Language Models (LLMs) for Knowledge Graph (KG) construction and reasoning. We engage in experiments across eight diverse datasets, focusing on four representative tasks encompassing entity and relation extraction, event extraction, link prediction, and question-answering, thereby thoroughly exploring LLMs' performance in the domain of construction and inference. Empirically, our findings suggest that LLMs, represented by GPT-4, are more suited as inference assistants rather than few-shot information extractors. Specifically, while GPT-4 exhibits good performance in tasks related to KG construction, it excels further in reasoning tasks, surpassing fine-tuned models in certain cases. Moreover, our investigation extends to the potential generalization ability of LLMs for information extraction, leading to the proposition of a Virtual Knowledge Extraction task and the development of the corresponding VINE dataset. Based on these empirical findings, we further propose AutoKG, a multi-agent-based approach employing LLMs and external sources for KG construction and reasoning. We anticipate that this research can provide invaluable insights for future undertakings in the field of knowledge graphs. The code and datasets are in https://github.com/zjunlp/AutoKG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17131v2">LLMsAgainstHate @ NLU of Devanagari Script Languages 2025: Hate Speech Detection and Target Identification in Devanagari Languages via Parameter Efficient Fine-Tuning of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      The detection of hate speech has become increasingly important in combating online hostility and its real-world consequences. Despite recent advancements, there is limited research addressing hate speech detection in Devanagari-scripted languages, where resources and tools are scarce. While large language models (LLMs) have shown promise in language-related tasks, traditional fine-tuning approaches are often infeasible given the size of the models. In this paper, we propose a Parameter Efficient Fine tuning (PEFT) based solution for hate speech detection and target identification. We evaluate multiple LLMs on the Devanagari dataset provided by (Thapa et al., 2025), which contains annotated instances in 2 languages - Hindi and Nepali. The results demonstrate the efficacy of our approach in handling Devanagari-scripted content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16844v3">Sim911: Towards Effective and Equitable 9-1-1 Dispatcher Training with an LLM-Enabled Simulation</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      Emergency response services are vital for enhancing public safety by safeguarding the environment, property, and human lives. As frontline members of these services, 9-1-1 dispatchers have a direct impact on response times and the overall effectiveness of emergency operations. However, traditional dispatcher training methods, which rely on role-playing by experienced personnel, are labor-intensive, time-consuming, and often neglect the specific needs of underserved communities. To address these challenges, we introduce Sim911, the first training simulation for 9-1-1 dispatchers powered by Large Language Models (LLMs). Sim911 enhances training through three key technical innovations: (1) knowledge construction, which utilizes archived 9-1-1 call data to generate simulations that closely mirror real-world scenarios; (2) context-aware controlled generation, which employs dynamic prompts and vector bases to ensure that LLM behavior aligns with training objectives; and (3) validation with looped correction, which filters out low-quality responses and refines the system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06175v3">Clustering Algorithms and RAG Enhancing Semi-Supervised Text Classification with Large LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-26
    </div>
    <details class="paper-abstract">
      This paper proposes a Clustering, Labeling, then Augmenting framework that significantly enhances performance in Semi-Supervised Text Classification (SSTC) tasks, effectively addressing the challenge of vast datasets with limited labeled examples. Unlike traditional SSTC approaches that rely on a predefined small set of labeled data to generate pseudo-labels for the unlabeled data, this framework innovatively employs clustering to select representative "landmarks" for labeling. These landmarks subsequently act as intermediaries in an ensemble of augmentation techniques, including Retrieval-Augmented Generation (RAG), Large Language Model (LLMs)-based rewriting, and synonym substitution, to generate synthetic labeled data without making pseudo-labels for the unlabeled data. Empirical results show that even in complex text document classification scenarios involving over 100 categories, our method achieves state-of-the-art accuracies of 95.41% on the Reuters dataset and 82.43% on the Web of Science dataset. Our approach significantly reduces the reliance on human labeling efforts and the associated expenses, while simultaneously ensuring high data quality and minimizing privacy risks. The finetuning results further show the efficiency of fine-tuning LLMs for text classification tasks, highlighting a robust solution for leveraging limited labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12642v2">Zero-shot Text-guided Infinite Image Synthesis with LLM guidance</a></div>
    <div class="paper-meta">
      📅 2024-12-26
      | 💬 This paper is being withdrawn due to issues of misconduct in the experiments presented in Table 2 and Figures 6, 7, and 8. We recognize this as an ethical concern and sincerely apologize to the research community for any inconvenience it may have caused
    </div>
    <details class="paper-abstract">
      Text-guided image editing and generation methods have diverse real-world applications. However, text-guided infinite image synthesis faces several challenges. First, there is a lack of text-image paired datasets with high-resolution and contextual diversity. Second, expanding images based on text requires global coherence and rich local context understanding. Previous studies have mainly focused on limited categories, such as natural landscapes, and also required to train on high-resolution images with paired text. To address these challenges, we propose a novel approach utilizing Large Language Models (LLMs) for both global coherence and local context understanding, without any high-resolution text-image paired training dataset. We train the diffusion model to expand an image conditioned on global and local captions generated from the LLM and visual feature. At the inference stage, given an image and a global caption, we use the LLM to generate a next local caption to expand the input image. Then, we expand the image using the global caption, generated local caption and the visual feature to consider global consistency and spatial local context. In experiments, our model outperforms the baselines both quantitatively and qualitatively. Furthermore, our model demonstrates the capability of text-guided arbitrary-sized image generation in zero-shot manner with LLM guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16145v2">Offline Reinforcement Learning for LLM Multi-Step Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Improving the multi-step reasoning ability of large language models (LLMs) with offline reinforcement learning (RL) is essential for quickly adapting them to complex tasks. While Direct Preference Optimization (DPO) has shown promise in aligning LLMs with human preferences, it is less suitable for multi-step reasoning tasks because (1) DPO relies on paired preference data, which is not readily available for multi-step reasoning tasks, and (2) it treats all tokens uniformly, making it ineffective for credit assignment in multi-step reasoning tasks, which often come with sparse reward. In this work, we propose OREO (Offline Reasoning Optimization), an offline RL method for enhancing LLM multi-step reasoning. Building on insights from previous works of maximum entropy reinforcement learning, it jointly learns a policy model and value function by optimizing the soft Bellman Equation. We show in principle that it reduces the need to collect pairwise data and enables better credit assignment. Empirically, OREO surpasses existing offline learning methods on multi-step reasoning benchmarks, including mathematical reasoning tasks (GSM8K, MATH) and embodied agent control (ALFWorld). The approach can be extended to a multi-iteration framework when additional resources are available. Furthermore, the learned value function can be leveraged to guide the tree search for free, which can further boost performance during test time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18934v1">Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 9 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Due to the high resource demands of Large Language Models (LLMs), achieving widespread deployment on consumer-grade devices presents significant challenges. Typically, personal or consumer-grade devices, including servers configured prior to the era of large-scale models, generally have relatively weak GPUs and relatively strong CPUs. However, most current methods primarily depend on GPUs for computation. Therefore, we propose Dovetail, an approach that deploys the draft model on the GPU to generate draft tokens while allowing the target model to perform parallel verification on the CPU, thereby improving the utilization of all available hardware resources and occupying less inter-device communication bandwidth. Accordingly, we have redesigned the draft model to better align with heterogeneous hardware characteristics. To this end, we implemented several optimizations: reducing the number of draft tokens to mitigate latency in parallel verification, increasing the depth of the draft model to enhance its predictive capacity, and introducing DGF (Dynamic Gating Fusion) to improve the integration of features and token embeddings. In the HumanEval benchmark, Dovetail achieved an inference speed of 5.86 tokens per second for LLaMA2-Chat-7B using 3GB of VRAM, representing an approximately 2.77x improvement over CPU-only inference. Furthermore, the inference speed was increased to 8 tokens per second when utilizing 7GB of VRAM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18925v1">HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      The breakthrough of OpenAI o1 highlights the potential of enhancing reasoning to improve LLM. Yet, most research in reasoning has focused on mathematical tasks, leaving domains like medicine underexplored. The medical domain, though distinct from mathematics, also demands robust reasoning to provide reliable answers, given the high standards of healthcare. However, verifying medical reasoning is challenging, unlike those in mathematics. To address this, we propose verifiable medical problems with a medical verifier to check the correctness of model outputs. This verifiable nature enables advancements in medical reasoning through a two-stage approach: (1) using the verifier to guide the search for a complex reasoning trajectory for fine-tuning LLMs, (2) applying reinforcement learning (RL) with verifier-based rewards to enhance complex reasoning further. Finally, we introduce HuatuoGPT-o1, a medical LLM capable of complex reasoning, which outperforms general and medical-specific baselines using only 40K verifiable problems. Experiments show complex reasoning improves medical problem-solving and benefits more from RL. We hope our approach inspires advancements in reasoning across medical and other specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06540v3">Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Scaling laws for large language models (LLMs) predict model performance based on parameters like size and training data. However, differences in training configurations and data processing across model families lead to significant variations in benchmark performance, making it difficult for a single scaling law to generalize across all LLMs. On the other hand, training family-specific scaling laws requires training models of varying sizes for every family. In this work, we propose Skills Scaling Laws (SSLaws, pronounced as Sloth), a novel scaling law that leverages publicly available benchmark data and assumes LLM performance is driven by low-dimensional latent skills, such as reasoning and instruction following. These latent skills are influenced by computational resources like model size and training tokens but with varying efficiencies across model families. Sloth exploits correlations across benchmarks to provide more accurate and interpretable predictions while alleviating the need to train multiple LLMs per family. We present both theoretical results on parameter identification and empirical evaluations on 12 prominent benchmarks, from Open LLM Leaderboard v1/v2, demonstrating that Sloth predicts LLM performance efficiently and offers insights into scaling behaviors for downstream tasks such as coding and emotional intelligence applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08582v3">On the Universal Truthfulness Hyperplane Inside LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 EMNLP 2024: Camera-ready version
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated remarkable abilities across various fields, hallucination remains a significant challenge. Recent studies have explored hallucinations through the lens of internal representations, proposing mechanisms to decipher LLMs' adherence to facts. However, these approaches often fail to generalize to out-of-distribution data, leading to concerns about whether internal representation patterns reflect fundamental factual awareness, or only overfit spurious correlations on the specific datasets. In this work, we investigate whether a universal truthfulness hyperplane that distinguishes the model's factually correct and incorrect outputs exists within the model. To this end, we scale up the number of training datasets and conduct an extensive evaluation -- we train the truthfulness hyperplane on a diverse collection of over 40 datasets and examine its cross-task, cross-domain, and in-domain generalization. Our results indicate that increasing the diversity of the training datasets significantly enhances the performance in all scenarios, while the volume of data samples plays a less critical role. This finding supports the optimistic hypothesis that a universal truthfulness hyperplane may indeed exist within the model, offering promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18377v2">ChaI-TeA: A Benchmark for Evaluating Autocompletion of Interactions with LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      The rise of LLMs has deflected a growing portion of human-computer interactions towards LLM-based chatbots. The remarkable abilities of these models allow users to interact using long, diverse natural language text covering a wide range of topics and styles. Phrasing these messages is a time and effort consuming task, calling for an autocomplete solution to assist users. We introduce the task of chatbot interaction autocomplete. We present ChaI-TeA: CHat InTEraction Autocomplete; An autcomplete evaluation framework for LLM-based chatbot interactions. The framework includes a formal definition of the task, coupled with suitable datasets and metrics. We use the framework to evaluate After formally defining the task along with suitable datasets and metrics, we test 9 models on the defined auto completion task, finding that while current off-the-shelf models perform fairly, there is still much room for improvement, mainly in ranking of the generated suggestions. We provide insights for practitioners working on this task and open new research directions for researchers in the field. We release our framework to serve as a foundation for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18811v1">DCIS: Efficient Length Extrapolation of LLMs via Divide-and-Conquer Scaling Factor Search</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based on the Transformer architecture usually have their context length limited due to the high training cost. Recent advancements extend the context window by adjusting the scaling factors of RoPE and fine-tuning. However, suboptimal initialization of these factors results in increased fine-tuning costs and reduced performance at target length. To address these challenges, we propose an innovative RoPE-based fine-tuning framework that diverges from conventional scaling factors search. Specifically, we present a Divide-and-Conquer Incremental Search (DCIS) algorithm that strategically determines the better scaling factors. Further fine-tuning with the identified scaling factors effectively extends the context window of LLMs. Empirical results demonstrate that our methodology not only mitigates performance decay at extended target lengths but also allows the model to fine-tune on short contexts and generalize to long contexts, thereby reducing the cost of fine-tuning. The scaling factors obtained through DCIS can even perform effectively without fine-tuning. Further analysis of the search space reveals that DCIS achieves twice the search efficiency compared to other methods. We also examine the impact of the non-strictly increasing scaling factors utilized in DCIS and evaluate the general capabilities of LLMs across various context lengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.12699v2">Explore the Potential of LLMs in Misinformation Detection: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-12-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have garnered significant attention for their powerful ability in natural language understanding and reasoning. In this paper, we present a comprehensive empirical study to explore the performance of LLMs on misinformation detection tasks. This study stands as the pioneering investigation into the understanding capabilities of multiple LLMs regarding both content and propagation across social media platforms. Our empirical studies on eight misinformation detection datasets show that LLM-based detectors can achieve comparable performance in text-based misinformation detection but exhibit notably constrained capabilities in comprehending propagation structure compared to existing models in propagation-based misinformation detection. Our experiments further demonstrate that LLMs exhibit great potential to enhance existing misinformation detection models. These findings highlight the potential ability of LLMs to detect misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14209v2">Agents4PLC: Automating Closed-loop PLC Code Generation and Verification in Industrial Control Systems using LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 12 pages (references included), 6 figures and 3 tables. New version updated with fixed site and github repo and along with some minor adjustements
    </div>
    <details class="paper-abstract">
      In industrial control systems, the generation and verification of Programmable Logic Controller (PLC) code are critical for ensuring operational efficiency and safety. While Large Language Models (LLMs) have made strides in automated code generation, they often fall short in providing correctness guarantees and specialized support for PLC programming. To address these challenges, this paper introduces Agents4PLC, a novel framework that not only automates PLC code generation but also includes code-level verification through an LLM-based multi-agent system. We first establish a comprehensive benchmark for verifiable PLC code generation area, transitioning from natural language requirements to human-written-verified formal specifications and reference PLC code. We further enhance our `agents' specifically for industrial control systems by incorporating Retrieval-Augmented Generation (RAG), advanced prompt engineering techniques, and Chain-of-Thought strategies. Evaluation against the benchmark demonstrates that Agents4PLC significantly outperforms previous methods, achieving superior results across a series of increasingly rigorous metrics. This research not only addresses the critical challenges in PLC programming but also highlights the potential of our framework to generate verifiable code applicable to real-world industrial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08336v1">Dynaseal: A Backend-Controlled LLM API Key Distribution Scheme with Constrained Invocation Parameters</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 5 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Due to the exceptional performance of Large Language Models (LLMs) in diverse downstream tasks,there has been an exponential growth in edge-device requests to cloud-based models.However, the current authentication mechanism using static Bearer Tokens in request headersfails to provide the flexibility and backend control required for edge-device deployments.To address these limitations, we propose Dynaseal,a novel methodology that enables fine-grained backend constraints on model invocations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00039v1">Speech Recognition With LLMs Adapted to Disordered Speech Using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-12-25
      | 💬 Accepted at ICASSP 2025
    </div>
    <details class="paper-abstract">
      We introduce a large language model (LLM) capable of processing speech inputs and show that tuning it further with reinforcement learning on human preference (RLHF) enables it to adapt better to disordered speech than traditional fine-tuning. Our method replaces low-frequency text tokens in an LLM's vocabulary with audio tokens and enables the model to recognize speech by fine-tuning it on speech with transcripts. We then use RL with rewards based on syntactic and semantic accuracy measures generalizing the LLM further to recognize disordered speech. While the resulting LLM does not outperform existing systems for speech recognition, we find that tuning with reinforcement learning using custom rewards leads to substantially better performance than supervised fine-tuning of the language model, specifically when adapting to speech in a different setting. This presents a compelling alternative tuning strategy for speech recognition using large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10179v2">Is Parameter Collision Hindering Continual Learning in LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from catastrophic forgetting when learning multiple tasks sequentially, making continual learning (CL) essential for their dynamic deployment. Existing state-of-the-art (SOTA) methods, such as O-LoRA, typically focus on constructing orthogonality tasks to decouple parameter interdependence from various domains.In this paper, we reveal that building non-collision parameters is a more critical factor in addressing CL challenges. Our theoretical and experimental analyses demonstrate that non-collision parameters can provide better task orthogonality, which is a sufficient but unnecessary condition. Furthermore, knowledge from multiple domains will be preserved in non-collision parameter subspaces, making it more difficult to forget previously seen data. Leveraging this insight, we propose Non-collision Low-Rank Adaptation (N-LoRA), a simple yet effective approach leveraging low collision rates to enhance CL in LLMs. Experimental results on multiple CL benchmarks indicate that N-LoRA achieves superior performance (+2.9), higher task orthogonality (*4.1 times), and lower parameter collision (*58.1 times) than SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06024v4">Thinking Assistants: LLM-Based Conversational Assistants that Help Users Think By Asking rather than Answering</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Many AI systems focus solely on providing solutions or explaining outcomes. However, complex tasks like research and strategic thinking often benefit from a more comprehensive approach to augmenting the thinking process rather than passively getting information. We introduce the concept of "Thinking Assistant", a new genre of assistants that help users improve decision-making with a combination of asking reflection questions based on expert knowledge. Through our lab study (N=80), these Large Language Model (LLM) based Thinking Assistants were better able to guide users to make important decisions, compared with conversational agents that only asked questions, provided advice, or neither. Based on the results, we develop a Thinking Assistant in academic career development, determining research trajectory or developing one's unique research identity, which requires deliberation, reflection and experts' advice accordingly. In a longitudinal deployment with 223 conversations, participants responded positively to approximately 65% of the responses. Our work proposes directions for developing more effective LLM agents. Rather than adhering to the prevailing authoritative approach of generating definitive answers, LLM agents aimed at assisting with cognitive enhancement should prioritize fostering reflection. They should initially provide responses designed to prompt thoughtful consideration through inquiring, followed by offering advice only after gaining a deeper understanding of the user's context and needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18702v1">CypherBench: Towards Precise Retrieval over Full-scale Modern Knowledge Graphs in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Retrieval from graph data is crucial for augmenting large language models (LLM) with both open-domain knowledge and private enterprise data, and it is also a key component in the recent GraphRAG system (edge et al., 2024). Despite decades of research on knowledge graphs and knowledge base question answering, leading LLM frameworks (e.g. Langchain and LlamaIndex) have only minimal support for retrieval from modern encyclopedic knowledge graphs like Wikidata. In this paper, we analyze the root cause and suggest that modern RDF knowledge graphs (e.g. Wikidata, Freebase) are less efficient for LLMs due to overly large schemas that far exceed the typical LLM context window, use of resource identifiers, overlapping relation types and lack of normalization. As a solution, we propose property graph views on top of the underlying RDF graph that can be efficiently queried by LLMs using Cypher. We instantiated this idea on Wikidata and introduced CypherBench, the first benchmark with 11 large-scale, multi-domain property graphs with 7.8 million entities and over 10,000 questions. To achieve this, we tackled several key challenges, including developing an RDF-to-property graph conversion engine, creating a systematic pipeline for text-to-Cypher task generation, and designing new evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v2">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      In this work, we investigate the causal reasoning abilities of large language models (LLMs) through the representative problem of inferring causal relationships from narratives. We find that even state-of-the-art language models rely on unreliable shortcuts, both in terms of the narrative presentation and their parametric knowledge. For example, LLMs tend to determine causal relationships based on the topological ordering of events (i.e., earlier events cause later ones), resulting in lower performance whenever events are not narrated in their exact causal order. Similarly, we demonstrate that LLMs struggle with long-term causal reasoning and often fail when the narratives are long and contain many events. Additionally, we show LLMs appear to rely heavily on their parametric knowledge at the expense of reasoning over the provided narrative. This degrades their abilities whenever the narrative opposes parametric knowledge. We extensively validate these failure modes through carefully controlled synthetic experiments, as well as evaluations on real-world narratives. Finally, we observe that explicitly generating a causal graph generally improves performance while naive chain-of-thought is ineffective. Collectively, our results distill precise failure modes of current state-of-the-art models and can pave the way for future techniques to enhance causal reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18695v1">TimelyLLM: Segmented LLM Serving System for Time-sensitive Robotic Applications</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4 and Llama3 can already comprehend complex commands and process diverse tasks. This advancement facilitates their application in controlling drones and robots for various tasks. However, existing LLM serving systems typically employ a first-come, first-served (FCFS) batching mechanism, which fails to address the time-sensitive requirements of robotic applications. To address it, this paper proposes a new system named TimelyLLM serving multiple robotic agents with time-sensitive requests. TimelyLLM introduces novel mechanisms of segmented generation and scheduling that optimally leverage redundancy between robot plan generation and execution phases. We report an implementation of TimelyLLM on a widely-used LLM serving framework and evaluate it on a range of robotic applications. Our evaluation shows that TimelyLLM improves the time utility up to 1.97x, and reduces the overall waiting time by 84%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18690v1">AgreeMate: Teaching LLMs to Haggle</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 15 pages, 22 figures, 6 tables
    </div>
    <details class="paper-abstract">
      We introduce AgreeMate, a framework for training Large Language Models (LLMs) to perform strategic price negotiations through natural language. We apply recent advances to a negotiation setting where two agents (i.e. buyer or seller) use natural language to bargain on goods using coarse actions. Specifically, we present the performance of Large Language Models when used as agents within a decoupled (modular) bargaining architecture. We demonstrate that using prompt engineering, fine-tuning, and chain-of-thought prompting enhances model performance, as defined by novel metrics. We use attention probing to show model attention to semantic relationships between tokens during negotiations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18588v1">A Paragraph is All It Takes: Rich Robot Behaviors from Interacting, Trusted LLMs</a></div>
    <div class="paper-meta">
      📅 2024-12-24
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are compact representations of all public knowledge of our physical environment and animal and human behaviors. The application of LLMs to robotics may offer a path to highly capable robots that perform well across most human tasks with limited or even zero tuning. Aside from increasingly sophisticated reasoning and task planning, networks of (suitably designed) LLMs offer ease of upgrading capabilities and allow humans to directly observe the robot's thinking. Here we explore the advantages, limitations, and particularities of using LLMs to control physical robots. The basic system consists of four LLMs communicating via a human language data bus implemented via web sockets and ROS2 message passing. Surprisingly, rich robot behaviors and good performance across different tasks could be achieved despite the robot's data fusion cycle running at only 1Hz and the central data bus running at the extremely limited rates of the human brain, of around 40 bits/s. The use of natural language for inter-LLM communication allowed the robot's reasoning and decision making to be directly observed by humans and made it trivial to bias the system's behavior with sets of rules written in plain English. These rules were immutably written into Ethereum, a global, public, and censorship resistant Turing-complete computer. We suggest that by using natural language as the data bus among interacting AIs, and immutable public ledgers to store behavior constraints, it is possible to build robots that combine unexpectedly rich performance, upgradability, and durable alignment with humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18573v1">How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      Recently, an increasing number of AI-driven programming assistants powered by code LLMs have been integrated into various real-world software development environments, significantly boosting developer productivity. However, existing code generation benchmarks primarily focus on general-purpose scenarios, leaving the code generation performance of LLMs for specific application domains largely unknown. In this paper, we introduce a new benchmark, MultiCodeBench, to fill this gap. MultiCodeBench comprises 2,400 programming tasks, covering 12 popular software development domains and 15 programming languages. Specifically, we perform in-depth research to identify these 12 application domains. Given that each domain may involve multiple technical frameworks, and that different frameworks present distinct challenges in the coding process, we categorize the commonly used frameworks and platforms within each domain. We then sample programming problems from GitHub repositories related to these subdomains. To ensure the quality of the tasks and mitigate data leakage issues, we invite annotators to rewrite the docstrings for each task in MultiCodeBench. Additionally, we build a static analysis-based dependency parsing tool to extract the dependencies in the ground truth for each task, enabling deeper performance analysis. Through extensive experiments on MultiCodeBench with eleven representative mainstream LLMs, we reveal the code generation performance of the LLMs across different application domains, providing practical insights for developers in downstream fields when selecting LLMs. Furthermore, we analyze the reasons behind the models' failures in completing software application development tasks, offering guidance for model developers to enhance domain-specific code generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18497v1">Think or Remember? Detecting and Directing LLMs Towards Memorization or Generalization</a></div>
    <div class="paper-meta">
      📅 2024-12-24
    </div>
    <details class="paper-abstract">
      In this paper, we explore the foundational mechanisms of memorization and generalization in Large Language Models (LLMs), inspired by the functional specialization observed in the human brain. Our investigation serves as a case study leveraging specially designed datasets and experimental-scale LLMs to lay the groundwork for understanding these behaviors. Specifically, we aim to first enable LLMs to exhibit both memorization and generalization by training with the designed dataset, then (a) examine whether LLMs exhibit neuron-level spatial differentiation for memorization and generalization, (b) predict these behaviors using model internal representations, and (c) steer the behaviors through inference-time interventions. Our findings reveal that neuron-wise differentiation of memorization and generalization is observable in LLMs, and targeted interventions can successfully direct their behavior.
    </details>
</div>
