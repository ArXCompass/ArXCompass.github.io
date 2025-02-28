# llm - 2025_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13055v1">LAMD: Context-driven Android Malware Detection and Classification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      The rapid growth of mobile applications has escalated Android malware threats. Although there are numerous detection methods, they often struggle with evolving attacks, dataset biases, and limited explainability. Large Language Models (LLMs) offer a promising alternative with their zero-shot inference and reasoning capabilities. However, applying LLMs to Android malware detection presents two key challenges: (1)the extensive support code in Android applications, often spanning thousands of classes, exceeds LLMs' context limits and obscures malicious behavior within benign functionality; (2)the structural complexity and interdependencies of Android applications surpass LLMs' sequence-based reasoning, fragmenting code analysis and hindering malicious intent inference. To address these challenges, we propose LAMD, a practical context-driven framework to enable LLM-based Android malware detection. LAMD integrates key context extraction to isolate security-critical code regions and construct program structures, then applies tier-wise code reasoning to analyze application behavior progressively, from low-level instructions to high-level semantics, providing final prediction and explanation. A well-designed factual consistency verification mechanism is equipped to mitigate LLM hallucinations from the first tier. Evaluation in real-world settings demonstrates LAMD's effectiveness over conventional detectors, establishing a feasible basis for LLM-driven malware analysis in dynamic threat landscapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18585v2">Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 1. We have updated the results for DeepSeek-R1, and all of our original conclusions remain valid. 2. Our proposed Tip approach remains effective in Best-of-N scenarios (e.g., self-consistency and Laconic Decoding) when built on DeepSeek-R1
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as OpenAI's o1 have demonstrated remarkable abilities in complex reasoning tasks by scaling test-time compute and exhibiting human-like deep thinking. However, we identify a phenomenon we term underthinking, where o1-like LLMs frequently switch between different reasoning thoughts without sufficiently exploring promising paths to reach a correct solution. This behavior leads to inadequate depth of reasoning and decreased performance, particularly on challenging mathematical problems. To systematically analyze this issue, we conduct experiments on three challenging test sets and two representative open-source o1-like models, revealing that frequent thought switching correlates with incorrect responses. We introduce a novel metric to quantify underthinking by measuring token efficiency in incorrect answers. To address underthinking, we propose a decoding strategy with thought switching penalty TIP that discourages premature transitions between thoughts, encouraging deeper exploration of each reasoning path. Experimental results demonstrate that our approach improves accuracy across challenging datasets without requiring model fine-tuning. Our findings contribute to understanding reasoning inefficiencies in o1-like LLMs and offer a practical solution to enhance their problem-solving capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13031v1">HPSS: Heuristic Prompting Strategy Search for LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 32 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Since the adoption of large language models (LLMs) for text evaluation has become increasingly prevalent in the field of natural language processing (NLP), a series of existing works attempt to optimize the prompts for LLM evaluators to improve their alignment with human judgment. However, their efforts are limited to optimizing individual factors of evaluation prompts, such as evaluation criteria or output formats, neglecting the combinatorial impact of multiple factors, which leads to insufficient optimization of the evaluation pipeline. Nevertheless, identifying well-behaved prompting strategies for adjusting multiple factors requires extensive enumeration. To this end, we comprehensively integrate 8 key factors for evaluation prompts and propose a novel automatic prompting strategy optimization method called Heuristic Prompting Strategy Search (HPSS). Inspired by the genetic algorithm, HPSS conducts an iterative search to find well-behaved prompting strategies for LLM evaluators. A heuristic function is employed to guide the search process, enhancing the performance of our algorithm. Extensive experiments across four evaluation tasks demonstrate the effectiveness of HPSS, consistently outperforming both human-designed evaluation prompts and existing automatic prompt optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16813v2">PeerArg: Argumentative Peer Review with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)
    </div>
    <details class="paper-abstract">
      Peer review is an essential process to determine the quality of papers submitted to scientific conferences or journals. However, it is subjective and prone to biases. Several studies have been conducted to apply techniques from NLP to support peer review, but they are based on black-box techniques and their outputs are difficult to interpret and trust. In this paper, we propose a novel pipeline to support and understand the reviewing and decision-making processes of peer review: the PeerArg system combining LLMs with methods from knowledge representation. PeerArg takes in input a set of reviews for a paper and outputs the paper acceptance prediction. We evaluate the performance of the PeerArg pipeline on three different datasets, in comparison with a novel end-2-end LLM that uses few-shot learning to predict paper acceptance given reviews. The results indicate that the end-2-end LLM is capable of predicting paper acceptance from reviews, but a variant of the PeerArg pipeline outperforms this LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13016v1">LLM-Powered Proactive Data Systems</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      With the power of LLMs, we now have the ability to query data that was previously impossible to query, including text, images, and video. However, despite this enormous potential, most present-day data systems that leverage LLMs are reactive, reflecting our community's desire to map LLMs to known abstractions. Most data systems treat LLMs as an opaque black box that operates on user inputs and data as is, optimizing them much like any other approximate, expensive UDFs, in conjunction with other relational operators. Such data systems do as they are told, but fail to understand and leverage what the LLM is being asked to do (i.e. the underlying operations, which may be error-prone), the data the LLM is operating on (e.g., long, complex documents), or what the user really needs. They don't take advantage of the characteristics of the operations and/or the data at hand, or ensure correctness of results when there are imprecisions and ambiguities. We argue that data systems instead need to be proactive: they need to be given more agency -- armed with the power of LLMs -- to understand and rework the user inputs and the data and to make decisions on how the operations and the data should be represented and processed. By allowing the data system to parse, rewrite, and decompose user inputs and data, or to interact with the user in ways that go beyond the standard single-shot query-result paradigm, the data system is able to address user needs more efficiently and effectively. These new capabilities lead to a rich design space where the data system takes more initiative: they are empowered to perform optimization based on the transformation operations, data characteristics, and user intent. We discuss various successful examples of how this framework has been and can be applied in real-world tasks, and present future directions for this ambitious research agenda.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13010v1">Adaptive Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Adaptive Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12988v1">Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought processes of a character. Using Lu Xun, a renowned Chinese writer, as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope that this work inspires future research on deep character persona simulation LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12982v1">Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 49 pages, 16 figures. Technical Report of Sailor2: https://sea-sailor.github.io/blog/sailor2/
    </div>
    <details class="paper-abstract">
      Sailor2 is a family of cutting-edge multilingual language models for South-East Asian (SEA) languages, available in 1B, 8B, and 20B sizes to suit diverse applications. Building on Qwen2.5, Sailor2 undergoes continuous pre-training on 500B tokens (400B SEA-specific and 100B replay tokens) to support 13 SEA languages while retaining proficiency in Chinese and English. Sailor2-20B model achieves a 50-50 win rate against GPT-4o across SEA languages. We also deliver a comprehensive cookbook on how to develop the multilingual model in an efficient manner, including five key aspects: data curation, pre-training, post-training, model customization and evaluation. We hope that Sailor2 model (Apache 2.0 license) will drive language development in the SEA region, and Sailor2 cookbook will inspire researchers to build more inclusive LLMs for other under-served languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22071v2">Distinguishing Ignorance from Error in LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are susceptible to hallucinations -- factually incorrect outputs -- leading to a large body of work on detecting and mitigating such cases. We argue that it is important to distinguish between two types of hallucinations: ones where the model does not hold the correct answer in its parameters, which we term HK-, and ones where the model answers incorrectly despite having the required knowledge, termed HK+. We first find that HK+ hallucinations are prevalent and occur across models and datasets. Then, we demonstrate that distinguishing between these two cases is beneficial for mitigating hallucinations. Importantly, we show that different models hallucinate on different examples, which motivates constructing model-specific hallucination datasets for training detectors. Overall, our findings draw attention to classifying types of hallucinations and provide means to handle them more effectively. The code is available at https://github.com/technion-cs-nlp/hallucination-mitigation .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12964v1">Trust Me, I'm Wrong: High-Certainty Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate outputs that lack grounding in real-world facts, a phenomenon known as hallucinations. Prior research has associated hallucinations with model uncertainty, leveraging this relationship for hallucination detection and mitigation. In this paper, we challenge the underlying assumption that all hallucinations are associated with uncertainty. Using knowledge detection and uncertainty measurement methods, we demonstrate that models can hallucinate with high certainty even when they have the correct knowledge. We further show that high-certainty hallucinations are consistent across models and datasets, distinctive enough to be singled out, and challenge existing mitigation methods. Our findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve mitigation strategies to enhance LLM safety. The code is available at https://github.com/technion-cs-nlp/Trust_me_Im_wrong .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12962v1">Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Limited by the context window size of Large Language Models(LLMs), handling various tasks with input tokens exceeding the upper limit has been challenging, whether it is a simple direct retrieval task or a complex multi-hop reasoning task. Although various methods have been proposed to enhance the long-context processing capabilities of LLMs, they either incur substantial post-training costs, or require additional tool modules(e.g.,RAG), or have not shown significant improvement in realistic tasks. Our work observes the correlation between the attention distribution and generated answers across each layer, and establishes the attention allocation aligns with retrieval-augmented capabilities through experiments. Drawing on the above insights, we propose a novel method InfiniRetri that leverages the LLMs's own attention information to enable accurate retrieval across inputs of infinitely length. Our evaluations indicate that InfiniRetri achieves 100% accuracy in the Needle-In-a-Haystack(NIH) test over 1M tokens using a 0.5B parameter model, surpassing other method or larger models and setting a new state-of-the-art(SOTA). Moreover, our method achieves significant performance improvements on real-world benchmarks, with a maximum 288% improvement. In addition, InfiniRetri can be applied to any Transformer-based LLMs without additional training and substantially reduces inference latency and compute overhead in long texts. In summary, our comprehensive studies show InfiniRetri's potential for practical applications and creates a paradigm for retrievaling information using LLMs own capabilities under infinite-length tokens. Code will be released in link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05193v2">RevisEval: Improving LLM-as-a-Judge via Response-Adapted References</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      With significant efforts in recent studies, LLM-as-a-Judge has become a cost-effective alternative to human evaluation for assessing text generation quality in a wide range of tasks. However, there still remains a reliability gap between LLM-as-a-Judge and human evaluation. One important reason is the lack of guided oracles in the evaluation process. Motivated by the role of reference pervasively used in classic text evaluation, we introduce RevisEval, a novel text generation evaluation paradigm via the response-adapted references. RevisEval is driven by the key observation that an ideal reference should maintain the necessary relevance to the response to be evaluated. Specifically, RevisEval leverages the text revision capabilities of large language models (LLMs) to adaptively revise the response, then treat the revised text as the reference (response-adapted reference) for the subsequent evaluation. Extensive experiments demonstrate that RevisEval outperforms traditional reference-free and reference-based evaluation paradigms that use LLM-as-a-Judge across NLG tasks and open-ended instruction-following tasks. More importantly, our response-adapted references can further boost the classical text metrics, e.g., BLEU and BERTScore, compared to traditional references and even rival the LLM-as-a-Judge. A detailed analysis is also conducted to confirm RevisEval's effectiveness in bias reduction, the impact of inference cost, and reference relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12945v1">LLMPopcorn: An Empirical Study of LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Popular Micro-videos, dominant on platforms like TikTok and YouTube, hold significant commercial value. The rise of high-quality AI-generated content has spurred interest in AI-driven micro-video creation. However, despite the advanced capabilities of large language models (LLMs) like ChatGPT and DeepSeek in text generation and reasoning, their potential to assist the creation of popular micro-videos remains largely unexplored. In this paper, we conduct an empirical study on LLM-assisted popular micro-video generation (LLMPopcorn). Specifically, we investigate the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? By exploring these questions, we show that advanced LLMs like DeepSeek-V3 enable micro-video generation to achieve popularity comparable to human-created content. Prompt enhancements further boost popularity, and benchmarking highlights DeepSeek-V3 and DeepSeek-R1 among LLMs, while LTX-Video and HunyuanVideo lead in video generation. This pioneering work advances AI-assisted micro-video creation, uncovering new research opportunities. We will release the code and datasets to support future studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12509v2">Can You Trust LLM Judgments? Reliability of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become increasingly powerful and ubiquitous, but their stochastic nature poses challenges to the reliability of their outputs. While deterministic settings can improve consistency, they do not guarantee reliability, as a single sample from the model's probability distribution can still be misleading. Building upon the concept of LLM-as-a-judge, we introduce a novel framework for rigorously evaluating the reliability of LLM judgments, leveraging McDonald's omega. We evaluate the reliability of LLMs when judging the outputs of other LLMs on standard single-turn and multi-turn benchmarks, simultaneously investigating the impact of temperature on reliability. By analyzing these results, we demonstrate the limitations of fixed randomness and the importance of considering multiple samples, which we show has significant implications for downstream applications. Our findings highlight the need for a nuanced understanding of LLM reliability and the potential risks associated with over-reliance on single-shot evaluations. This work provides a crucial step towards building more trustworthy and reliable LLM-based systems and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15175v3">ToxiLab: How Well Do Open-Source LLMs Generate Synthetic Toxicity Data?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Effective toxic content detection relies heavily on high-quality and diverse data, which serve as the foundation for robust content moderation models. Synthetic data has become a common approach for training models across various NLP tasks. However, its effectiveness remains uncertain for highly subjective tasks like hate speech detection, with previous research yielding mixed results. This study explores the potential of open-source LLMs for harmful data synthesis, utilizing controlled prompting and supervised fine-tuning techniques to enhance data quality and diversity. We systematically evaluated 6 open source LLMs on 5 datasets, assessing their ability to generate diverse, high-quality harmful data while minimizing hallucination and duplication. Our results show that Mistral consistently outperforms other open models, and supervised fine-tuning significantly enhances data reliability and diversity. We further analyze the trade-offs between prompt-based vs. fine-tuned toxic data synthesis, discuss real-world deployment challenges, and highlight ethical considerations. Our findings demonstrate that fine-tuned open source LLMs provide scalable and cost-effective solutions to augment toxic content detection datasets, paving the way for more accessible and transparent content moderation tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12929v1">Flow-of-Options: Diversified and Improved LLM Reasoning by Thinking Through Options</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Github code: https://github.com/flagshippioneering/Flow-of-Options
    </div>
    <details class="paper-abstract">
      We present a novel reasoning approach called Flow-of-Options (FoO), designed to address intrinsic biases in Large Language Models (LLMs). FoO enables LLMs to systematically explore a diverse range of possibilities in their reasoning, as demonstrated by an FoO-based agentic system for autonomously solving Machine Learning tasks (AutoML). Our framework outperforms state-of-the-art baselines, achieving improvements of 38.2% - 69.2% on standard data science tasks, and 37.4% - 47.9% on therapeutic chemistry tasks. With an overall operation cost under $1 per task, our framework is well-suited for cost-sensitive applications. Beyond classification and regression, we illustrate the broader applicability of our FoO-based agentic system to tasks such as reinforcement learning and image generation. Our framework presents significant advancements compared to current state-of-the-art agentic systems for AutoML, due to the benefits of FoO in enforcing diversity in LLM solutions through compressed, explainable representations that also support long-term memory when combined with case-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12924v1">Conditioning LLMs to Generate Code-Switched Text: A Methodology Grounded in Naturally Occurring Data</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. This paper presents a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12923v1">On-Device LLMs for Home Assistant: Dual Role in Intent Detection and Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      This paper investigates whether Large Language Models (LLMs), fine-tuned on synthetic but domain-representative data, can perform the twofold task of (i) slot and intent detection and (ii) natural language response generation for a smart home assistant, while running solely on resource-limited, CPU-only edge hardware. We fine-tune LLMs to produce both JSON action calls and text responses. Our experiments show that 16-bit and 8-bit quantized variants preserve high accuracy on slot and intent detection and maintain strong semantic coherence in generated text, while the 4-bit model, while retaining generative fluency, suffers a noticeable drop in device-service classification accuracy. Further evaluations on noisy human (non-synthetic) prompts and out-of-domain intents confirm the models' generalization ability, obtaining around 80--86\% accuracy. While the average inference time is 5--6 seconds per query -- acceptable for one-shot commands but suboptimal for multi-turn dialogue -- our results affirm that an on-device LLM can effectively unify command interpretation and flexible response generation for home automation without relying on specialized hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12918v1">Query Rewriting via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Query rewriting is a classical technique for transforming complex declarative SQL queries into ``lean'' equivalents that are conducive to (a) faster execution from a performance perspective, and (b) better understanding from a developer perspective. The rewriting is typically achieved via transformation rules, but these rules are limited in scope and difficult to update in a production system. In recent times, LLM-based techniques have also been mooted, but they are prone to both semantic and syntactic errors. We investigate here, how the remarkable cognitive capabilities of LLMs can be leveraged for performant query rewriting while incorporating safeguards and optimizations to ensure correctness and efficiency. Our study shows that these goals can be progressively achieved through incorporation of (a) an ensemble suite of basic prompts, (b) database-sensitive prompts via redundancy removal and selectivity-based rewriting rules, and (c) LLM token probability-guided rewrite paths. Further, a suite of statistical and logic-based tools can be used to guard against errors produced by the model. We have implemented the above LLM-infused techniques in the LITHE system, and evaluated complex analytic queries from multiple benchmarks on contemporary database platforms. The results show significant improvements over SOTA rewriting techniques -- for instance, on TPC-DS, LITHE constructed productive (>1.5x speedup) rewrites for \emph{two-thirds} of the query suite, delivering four times more coverage than SOTA. Further, the geometric mean of its estimated execution speedups was an \emph{order-of-magnitude} jump over SOTA performance. In essence, LITHE offers a potent and robust LLM-based intermediary between enterprise applications and database engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12913v1">GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to FP16-based fine-tuning while significantly reducing memory usage (50%). Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12911v1">Knapsack Optimization-based Schema Linking for LLM-based Text-to-SQL Generation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Generating SQLs from user queries is a long-standing challenge, where the accuracy of initial schema linking significantly impacts subsequent SQL generation performance. However, current schema linking models still struggle with missing relevant schema elements or an excess of redundant ones. A crucial reason for this is that commonly used metrics, recall and precision, fail to capture relevant element missing and thus cannot reflect actual schema linking performance. Motivated by this, we propose an enhanced schema linking metric by introducing a restricted missing indicator. Accordingly, we introduce Knapsack optimization-based Schema Linking Agent (KaSLA), a plug-in schema linking agent designed to prevent the missing of relevant schema elements while minimizing the inclusion of redundant ones. KaSLA employs a hierarchical linking strategy that first identifies the optimal table linking and subsequently links columns within the selected table to reduce linking candidate space. In each linking process, it utilize a knapsack optimization approach to link potentially relevant elements while accounting for a limited tolerance of potential redundant ones.With this optimization, KaSLA-1.6B achieves superior schema linking results compared to large-scale LLMs, including deepseek-v3 with state-of-the-art (SOTA) schema linking method. Extensive experiments on Spider and BIRD benchmarks verify that KaSLA can significantly improve the SQL generation performance of SOTA text-to-SQL models by substituting their schema linking processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12904v1">Fraud-R1 : A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      We introduce Fraud-R1, a benchmark designed to evaluate LLMs' ability to defend against internet fraud and phishing in dynamic, real-world scenarios. Fraud-R1 comprises 8,564 fraud cases sourced from phishing scams, fake job postings, social media, and news, categorized into 5 major fraud types. Unlike previous benchmarks, Fraud-R1 introduces a multi-round evaluation pipeline to assess LLMs' resistance to fraud at different stages, including credibility building, urgency creation, and emotional manipulation. Furthermore, we evaluate 15 LLMs under two settings: 1. Helpful-Assistant, where the LLM provides general decision-making assistance, and 2. Role-play, where the model assumes a specific persona, widely used in real-world agent-based interactions. Our evaluation reveals the significant challenges in defending against fraud and phishing inducement, especially in role-play settings and fake job postings. Additionally, we observe a substantial performance gap between Chinese and English, underscoring the need for improved multilingual fraud detection capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12900v1">Soundwave: Less is More for Speech-Text Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Existing end-to-end speech large language models (LLMs) usually rely on large-scale annotated data for training, while data-efficient training has not been discussed in depth. We focus on two fundamental problems between speech and text: the representation space gap and sequence length inconsistency. We propose Soundwave, which utilizes an efficient training strategy and a novel architecture to address these issues. Results show that Soundwave outperforms the advanced Qwen2-Audio in speech translation and AIR-Bench speech tasks, using only one-fiftieth of the training data. Further analysis shows that Soundwave still retains its intelligence during conversation. The project is available at https://github.com/FreedomIntelligence/Soundwave.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v1">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12884v1">How desirable is alignment between LLMs and linguistically diverse human users?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      We discuss how desirable it is that Large Language Models (LLMs) be able to adapt or align their language behavior with users who may be diverse in their language use. User diversity may come about among others due to i) age differences; ii) gender characteristics, and/or iii) multilingual experience, and associated differences in language processing and use. We consider potential consequences for usability, communication, and LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15173v4">Second-Order Fine-Tuning without Pain for LLMs:A Hessian Informed Zeroth-Order Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with classic first-order optimizers entails prohibitive GPU memory due to the backpropagation process. Recent works have turned to zeroth-order optimizers for fine-tuning, which save substantial memory by using two forward passes. However, these optimizers are plagued by the heterogeneity of parameter curvatures across different dimensions. In this work, we propose HiZOO, a diagonal Hessian informed zeroth-order optimizer which is the first work to leverage the diagonal Hessian to enhance zeroth-order optimizer for fine-tuning LLMs. What's more, HiZOO avoids the expensive memory cost and only increases one forward pass per step. Extensive experiments on various models (350M~66B parameters) indicate that HiZOO improves model convergence, significantly reducing training steps and effectively enhancing model accuracy. Moreover, we visualize the optimization trajectories of HiZOO on test functions, illustrating its effectiveness in handling heterogeneous curvatures. Lastly, we provide theoretical proofs of convergence for HiZOO. Code is publicly available at https://anonymous.4open.science/r/HiZOO27F8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12853v1">S$^2$R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated the effectiveness of LLM test-time scaling. However, existing approaches to incentivize LLMs' deep thinking abilities generally require large-scale data or significant training efforts. Meanwhile, it remains unclear how to improve the thinking abilities of less powerful base models. In this work, we introduce S$^2$R, an efficient framework that enhances LLM reasoning by teaching models to self-verify and self-correct during inference. Specifically, we first initialize LLMs with iterative self-verification and self-correction behaviors through supervised fine-tuning on carefully curated data. The self-verification and self-correction skills are then further strengthened by both outcome-level and process-level reinforcement learning, with minimized resource requirements, enabling the model to adaptively refine its reasoning process during inference. Our results demonstrate that, with only 3.1k self-verifying and self-correcting behavior initialization samples, Qwen2.5-math-7B achieves an accuracy improvement from 51.0\% to 81.6\%, outperforming models trained on an equivalent amount of long-CoT distilled data. Extensive experiments and analysis based on three base models across both in-domain and out-of-domain benchmarks validate the effectiveness of S$^2$R. Our code and data are available at https://github.com/NineAbyss/S2R.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12842v1">Towards Adaptive Feedback with AI: Comparing the Feedback Quality of LLMs and Teachers on Experimentation Protocols</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 This work has been submitted to the IJAIED for possible publication
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for fostering students' success in scientific inquiry. With advancements in artificial intelligence, large language models (LLMs) offer new possibilities for delivering instant and adaptive feedback. However, this feedback often lacks the pedagogical validation provided by real-world practitioners. To address this limitation, our study evaluates and compares the feedback quality of LLM agents with that of human teachers and science education experts on student-written experimentation protocols. Four blinded raters, all professionals in scientific inquiry and science education, evaluated the feedback texts generated by 1) the LLM agent, 2) the teachers and 3) the science education experts using a five-point Likert scale based on six criteria of effective feedback: Feed Up, Feed Back, Feed Forward, Constructive Tone, Linguistic Clarity, and Technical Terminology. Our results indicate that LLM-generated feedback shows no significant difference to that of teachers and experts in overall quality. However, the LLM agent's performance lags in the Feed Back dimension, which involves identifying and explaining errors within the student's work context. Qualitative analysis highlighted the LLM agent's limitations in contextual understanding and in the clear communication of specific errors. Our findings suggest that combining LLM-generated feedback with human expertise can enhance educational practices by leveraging the efficiency of LLMs and the nuanced understanding of educators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12836v1">An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing healthcare by improving diagnosis, patient care, and decision support through interactive communication. More recently, they have been applied to analyzing physiological time-series like wearable data for health insight extraction. Existing methods embed raw numerical sequences directly into prompts, which exceeds token limits and increases computational costs. Additionally, some studies integrated features extracted from time-series in textual prompts or applied multimodal approaches. However, these methods often produce generic and unreliable outputs due to LLMs' limited analytical rigor and inefficiency in interpreting continuous waveforms. In this paper, we develop an LLM-powered agent for physiological time-series analysis aimed to bridge the gap in integrating LLMs with well-established analytical tools. Built on the OpenCHA, an open-source LLM-powered framework, our agent features an orchestrator that integrates user interaction, data sources, and analytical tools to generate accurate health insights. To evaluate its effectiveness, we implement a case study on heart rate (HR) estimation from Photoplethysmogram (PPG) signals using a dataset of PPG and Electrocardiogram (ECG) recordings in a remote health monitoring study. The agent's performance is benchmarked against OpenAI GPT-4o-mini and GPT-4o, with ECG serving as the gold standard for HR estimation. Results demonstrate that our agent significantly outperforms benchmark models by achieving lower error rates and more reliable HR estimations. The agent implementation is publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11409v5">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Penetration testing, an essential component of software security testing, allows organizations to identify and remediate vulnerabilities in their systems, thus bolstering their defense mechanisms against cyberattacks. One recent advancement in the realm of penetration testing is the utilization of Language Models (LLMs). We explore the intersection of LLMs and penetration testing to gain insight into their capabilities and challenges in the context of privilege escalation. We introduce a fully automated privilege-escalation tool designed for evaluating the efficacy of LLMs for (ethical) hacking, executing benchmarks using multiple LLMs, and investigating their respective results. Our results show that GPT-4-turbo is well suited to exploit vulnerabilities (33-83% of vulnerabilities). GPT-3.5-turbo can abuse 16-50% of vulnerabilities, while local models, such as Llama3, can only exploit between 0 and 33% of the vulnerabilities. We analyze the impact of different context sizes, in-context learning, optional high-level guidance mechanisms, and memory management techniques. We discuss challenging areas for LLMs, including maintaining focus during testing, coping with errors, and finally comparing LLMs with human hackers. The current version of the LLM-guided privilege-escalation prototype can be found at https://github.com/ipa-labs/hackingBuddyGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08115v2">Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based multi-agent systems (MAS) show remarkable potential in collaborative problem-solving, yet they still face critical challenges: low communication efficiency, poor scalability, and a lack of effective parameter-updating optimization methods. We present Optima, a novel framework that addresses these issues by significantly enhancing both communication efficiency and task effectiveness in LLM-based MAS through LLM training. Optima employs an iterative generate, rank, select, and train paradigm with a reward function balancing task performance, token efficiency, and communication readability. We explore various RL algorithms, including Supervised Fine-Tuning, Direct Preference Optimization, and their hybrid approaches, providing insights into their effectiveness-efficiency trade-offs. We integrate Monte Carlo Tree Search-inspired techniques for DPO data generation, treating conversation turns as tree nodes to explore diverse interaction paths. Evaluated on common multi-agent tasks, including information-asymmetric question answering and complex reasoning, Optima shows consistent and substantial improvements over single-agent baselines and vanilla MAS based on Llama 3 8B, achieving up to 2.8x performance gain with less than 10\% tokens on tasks requiring heavy information exchange. Moreover, Optima's efficiency gains open new possibilities for leveraging inference-compute more effectively, leading to improved inference-time scaling laws. By addressing fundamental challenges in LLM-based MAS, Optima shows the potential towards scalable, efficient, and effective MAS (https://chenweize1998.github.io/optima-project-page).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05590v3">NYU CTF Bench: A Scalable Open-Source Benchmark Dataset for Evaluating LLMs in Offensive Security</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being deployed across various domains today. However, their capacity to solve Capture the Flag (CTF) challenges in cybersecurity has not been thoroughly evaluated. To address this, we develop a novel method to assess LLMs in solving CTF challenges by creating a scalable, open-source benchmark database specifically designed for these applications. This database includes metadata for LLM testing and adaptive learning, compiling a diverse range of CTF challenges from popular competitions. Utilizing the advanced function calling capabilities of LLMs, we build a fully automated system with an enhanced workflow and support for external tool calls. Our benchmark dataset and automated framework allow us to evaluate the performance of five LLMs, encompassing both black-box and open-source models. This work lays the foundation for future research into improving the efficiency of LLMs in interactive cybersecurity tasks and automated task planning. By providing a specialized benchmark, our project offers an ideal platform for developing, testing, and refining LLM-based approaches to vulnerability detection and resolution. Evaluating LLMs on these challenges and comparing with human performance yields insights into their potential for AI-driven cybersecurity solutions to perform real-world threat management. We make our benchmark dataset open source to public https://github.com/NYU-LLM-CTF/NYU_CTF_Bench along with our playground automated framework https://github.com/NYU-LLM-CTF/llm_ctf_automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04563v2">WaferLLM: A Wafer-Scale LLM Inference System</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Emerging AI accelerators increasingly adopt wafer-scale manufacturing technologies, integrating hundreds of thousands of AI cores in a mesh-based architecture with large distributed on-chip memory (tens of GB in total) and ultra-high on-chip memory bandwidth (tens of PB/s). However, current LLM inference systems, optimized for shared memory architectures like GPUs, fail to fully exploit these accelerators. We introduce WaferLLM, the first wafer-scale LLM inference system. WaferLLM is guided by a novel PLMR model (pronounced as "Plummer") that captures the unique hardware characteristics of wafer-scale architectures. Leveraging this model, WaferLLM pioneers wafer-scale LLM parallelism, optimizing the utilization of hundreds of thousands of on-chip cores. It also introduces MeshGEMM and MeshGEMV, the first GEMM and GEMV implementations designed to scale effectively on wafer-scale accelerators. Evaluations show that WaferLLM achieves 200$\times$ better wafer-scale accelerator utilization than state-of-the-art systems. On a commodity wafer-scale accelerator, WaferLLM delivers 606$\times$ faster and 22$\times$ more energy-efficient GEMV compared to an advanced GPU. For LLMs, based on 16-bit data type, WaferLLM achieves 2700 toks/sec/req decode speed on Llama3-8B model and 840 toks/sec/req decode speed on Qwen2-72B model, which enables 39$\times$ faster decoding with 1.7$\times$ better energy efficiency. We anticipate these numbers will grow significantly as wafer-scale AI models, software, and hardware continue to mature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02089v2">RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Add repair model ablation, update related work
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deployed as agents solve user-specified tasks over multiple steps while keeping the required manual engagement to a minimum. Crucially, such LLMs need to ground their generations in any feedback obtained to reliably achieve the desired outcomes. We propose an end-to-end reinforcement learning method for teaching models to leverage execution feedback in the realm of code synthesis, where state-of-the-art LLMs struggle to improve code iteratively compared to independent sampling. We benchmark on competitive programming tasks, where we achieve new state-of-the art results with both small (8B parameters) and large (70B) models while reducing the amount of samples required by an order of magnitude. Our analysis of inference-time behavior demonstrates that our method produces LLMs that effectively leverage automatic feedback over multiple steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05551v3">FRAME: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced human language understanding and generation, with pretraining data quality and organization being crucial to their performance. Multi-stage pretraining is a promising approach, but existing methods often lack quantitative criteria for data partitioning and instead rely on intuitive heuristics. In this paper, we propose the novel Four-quadRAnt Multi-stage prEtraining strategy (FRAME), guided by the established principle of organizing the pretraining process into four stages to achieve significant loss reductions four times. This principle is grounded in two key findings: first, training on high Perplexity (PPL) data followed by low PPL data, and second, training on low PPL difference (PD) data followed by high PD data, both causing the loss to drop significantly twice and performance enhancements. By partitioning data into four quadrants and strategically organizing them, FRAME achieves a remarkable 16.8% average improvement over random across MMLU and CMMLU for the 3B model, effectively boosting LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12769v1">How Much Do LLMs Hallucinate across Languages? On Multilingual Estimation of LLM Hallucination in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      In the age of misinformation, hallucination -- the tendency of Large Language Models (LLMs) to generate non-factual or unfaithful responses -- represents the main risk for their global utility. Despite LLMs becoming increasingly multilingual, the vast majority of research on detecting and quantifying LLM hallucination are (a) English-centric and (b) focus on machine translation (MT) and summarization, tasks that are less common ``in the wild'' than open information seeking. In contrast, we aim to quantify the extent of LLM hallucination across languages in knowledge-intensive long-form question answering. To this end, we train a multilingual hallucination detection model and conduct a large-scale study across 30 languages and 6 open-source LLM families. We start from an English hallucination detection dataset and rely on MT to generate (noisy) training data in other languages. We also manually annotate gold data for five high-resource languages; we then demonstrate, for these languages, that the estimates of hallucination rates are similar between silver (LLM-generated) and gold test sets, validating the use of silver data for estimating hallucination rates for other languages. For the final rates estimation, we build a knowledge-intensive QA dataset for 30 languages with LLM-generated prompts and Wikipedia articles as references. We find that, while LLMs generate longer responses with more hallucinated tokens for higher-resource languages, there is no correlation between length-normalized hallucination rates of languages and their digital representation. Further, we find that smaller LLMs exhibit larger hallucination rates than larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12743v1">"I know myself better, but not really greatly": Using LLMs to Detect and Explain LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in generating human-like texts, but the potential misuse of such LLM-generated texts raises the need to distinguish between human-generated and LLM-generated content. This paper explores the detection and explanation capabilities of LLM-based detectors of LLM-generated texts, in the context of a binary classification task (human-generated texts vs LLM-generated texts) and a ternary classification task (human-generated texts, LLM-generated texts, and undecided). By evaluating on six close/open-source LLMs with different sizes, our findings reveal that while self-detection consistently outperforms cross-detection, i.e., LLMs can detect texts generated by themselves more accurately than those generated by other LLMs, the performance of self-detection is still far from ideal, indicating that further improvements are needed. We also show that extending the binary to the ternary classification task with a new class "Undecided" can enhance both detection accuracy and explanation quality, with improvements being statistically significant and consistent across all LLMs. We finally conducted comprehensive qualitative and quantitative analyses on the explanation errors, which are categorized into three types: reliance on inaccurate features (the most frequent error), hallucinations, and incorrect reasoning. These findings with our human-annotated dataset emphasize the need for further research into improving both self-detection and self-explanation, particularly to address overfitting issues that may hinder generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14871v2">I don't trust you (anymore)! -- The effect of students' LLM use on Lecturer-Student-Trust in Higher Education</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Trust plays a pivotal role in Lecturer-Student-Collaboration, encompassing teaching and research aspects. The advent of Large Language Models (LLMs) in platforms like Open AI's ChatGPT, coupled with their cost-effectiveness and high-quality results, has led to their rapid adoption among university students. However, discerning genuine student input from LLM-generated output poses a challenge for lecturers. This dilemma jeopardizes the trust relationship between lecturers and students, potentially impacting university downstream activities, particularly collaborative research initiatives. Despite attempts to establish guidelines for student LLM use, a clear framework mutually beneficial for lecturers and students in higher education remains elusive. This study addresses the research question: How does the use of LLMs by students impact Informational and Procedural Justice, influencing Team Trust and Expected Team Performance? Methodically, we applied a quantitative construct-based survey, evaluated using techniques of Structural Equation Modelling (PLS- SEM) to examine potential relationships among these constructs. Our findings based on 23 valid respondents from Ndejje University indicate that lecturers are less concerned about the fairness of LLM use per se but are more focused on the transparency of student utilization, which significantly influences Team Trust positively. This research contributes to the global discourse on integrating and regulating LLMs and subsequent models in education. We propose that guidelines should support LLM use while enforcing transparency in Lecturer-Student- Collaboration to foster Team Trust and Performance. The study contributes valuable insights for shaping policies enabling ethical and transparent LLMs usage in education to ensure effectiveness of collaborative learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11006v2">Effective Self-Mining of In-Context Examples for Unsupervised Machine Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Accepted at NAACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance on a wide range of natural language processing (NLP) tasks, primarily through in-context learning (ICL). In ICL, the LLM is provided with examples that represent a given task such that it learns to generate answers for test inputs. However, access to these in-context examples is not guaranteed especially for low-resource or massively multilingual tasks. In this work, we propose an unsupervised approach to mine in-context examples for machine translation (MT), enabling unsupervised MT (UMT) across different languages. Our approach begins with word-level mining to acquire word translations that are then used to perform sentence-level mining. As the quality of mined parallel pairs may not be optimal due to noise or mistakes, we introduce a filtering criterion to select the optimal in-context examples from a pool of unsupervised parallel sentences. We evaluate our approach using two multilingual LLMs on 288 directions from the FLORES-200 dataset and analyze the impact of various linguistic features on performance. Our findings demonstrate the effectiveness of our unsupervised approach in mining in-context examples for MT, leading to better or comparable translation performance as translation with regular in-context samples (extracted from human-annotated data), while also outperforming the other state-of-the-art UMT methods by an average of $7$ BLEU points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07745v3">StepTool: Enhancing Multi-Step Tool Usage in LLMs through Step-Grained Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Ongoning Work
    </div>
    <details class="paper-abstract">
      Despite powerful text generation capabilities, large language models (LLMs) still need to learn how to utilize external tools to solve complex tasks, a process known as tool learning. Existing methods primarily rely on supervised fine-tuning to enhance tool-use capabilities, treating tool learning as a text-generation task while overlooking the decision-making complexities inherent in multi-step contexts. In this work, we propose modeling tool learning as a dynamic decision-making task and introduce StepTool, a novel step-grained reinforcement learning framework that enhances the multi-step tool use capabilities of LLMs. StepTool consists of two main components: Step-grained Reward Shaping, which assigns rewards at each tool interaction based on the success of tool invocation and its contribution to the task; and Step-grained Optimization, which uses policy gradient methods to optimize the model in a multi-step manner. Experimental results demonstrate that StepTool significantly outperforms existing methods in multi-step, tool-based tasks, offering a robust solution for tool learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10674v2">Can Multimodal LLMs do Visual Temporal Understanding and Reasoning? The answer is No!</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Our dataset can be found at \url{https://huggingface.co/datasets/fazliimam/temporal-vqa}
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have achieved significant advancements in tasks like Visual Question Answering (VQA) by leveraging foundational Large Language Models (LLMs). However, their abilities in specific areas such as visual temporal understanding, which is crucial for comprehending real-world dynamics, remain underexplored. To address this, we propose a challenging evaluation benchmark named TemporalVQA, consisting of two parts: 1) Temporal Order Understanding and 2) Time-lapse Estimation. The first part requires MLLMs to determine the sequence of events by analyzing temporally consecutive video frames. The second part presents image pairs with varying time differences, framed as multiple-choice questions, asking MLLMs to estimate the time-lapse between images with options ranging from seconds to years. Our evaluations of advanced MLLMs, including models like GPT-4o and Gemini-1.5-Pro, reveal significant challenges: GPT-4o achieved only 49.1% average consistent accuracy in temporal order task and 70% in time-lapse estimation, with open-source models performing even poorly. These findings underscore the limitations of current MLLMs in visual temporal understanding and reasoning, highlighting the need for further improvements for their temporal capability. Our dataset can be found at https://huggingface.co/datasets/fazliimam/temporal-vqa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12658v1">R.R.: Unveiling LLM Training Privacy through Recollection and Ranking</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLM's training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identical performance compared to baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release the replicate package of R.R. at a link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01269v5">CPRM: A LLM-based Continual Pre-training Framework for Relevance Modeling in Commercial Search</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 NAACL 2025
    </div>
    <details class="paper-abstract">
      Relevance modeling between queries and items stands as a pivotal component in commercial search engines, directly affecting the user experience. Given the remarkable achievements of large language models (LLMs) in various natural language processing (NLP) tasks, LLM-based relevance modeling is gradually being adopted within industrial search systems. Nevertheless, foundational LLMs lack domain-specific knowledge and do not fully exploit the potential of in-context learning. Furthermore, structured item text remains underutilized, and there is a shortage in the supply of corresponding queries and background knowledge. We thereby propose CPRM (Continual Pre-training for Relevance Modeling), a framework designed for the continual pre-training of LLMs to address these issues. Our CPRM framework includes three modules: 1) employing both queries and multi-field item to jointly pre-train for enhancing domain knowledge, 2) applying in-context pre-training, a novel approach where LLMs are pre-trained on a sequence of related queries or items, and 3) conducting reading comprehension on items to produce associated domain knowledge and background information (e.g., generating summaries and corresponding queries) to further strengthen LLMs. Results on offline experiments and online A/B testing demonstrate that our model achieves convincing performance compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12499v2">With a Grain of SALT: Are LLMs Fair Across Social Dimensions?</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      This paper presents a systematic analysis of biases in open-source Large Language Models (LLMs), across gender, religion, and race. Our study evaluates bias in smaller-scale Llama and Gemma models using the SALT ($\textbf{S}$ocial $\textbf{A}$ppropriateness in $\textbf{L}$LM-Generated $\textbf{T}$ext) dataset, which incorporates five distinct bias triggers: General Debate, Positioned Debate, Career Advice, Problem Solving, and CV Generation. To quantify bias, we measure win rates in General Debate and the assignment of negative roles in Positioned Debate. For real-world use cases, such as Career Advice, Problem Solving, and CV Generation, we anonymize the outputs to remove explicit demographic identifiers and use DeepSeek-R1 as an automated evaluator. We also address inherent biases in LLM-based evaluation, including evaluation bias, positional bias, and length bias, and validate our results through human evaluations. Our findings reveal consistent polarization across models, with certain demographic groups receiving systematically favorable or unfavorable treatment. By introducing SALT, we provide a comprehensive benchmark for bias analysis and underscore the need for robust bias mitigation strategies in the development of equitable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10858v2">Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Deep iterative chain-of-thought (CoT) reasoning enables LLMs to tackle complex tasks by progressively activating relevant pre-trained knowledge. However, it faces challenges in ensuring continual improvement and determining a stopping criterion. In this paper, we investigate whether the relevant knowledge that contributes directly to solving the given question can be activated from the initial reasoning path, thus circumventing the need for iterative refinement. Our experiments reveal that increasing the diversity of initial reasoning paths can achieve comparable or superior performance, a concept we term \textit{breadth reasoning}. However, existing breadth reasoning approaches, such as self-consistency, offer limited diversity. To address this limitation, we propose a simple yet effective method that enhances reasoning breadth by integrating contextual exploration with reduced sampling randomness. Extensive experiments demonstrate that our approach significantly outperforms deep iterative reasoning. Our code is provided in https://github.com/zongqianwu/breadth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12598v1">Bring Your Own Knowledge: A Survey of Methods for LLM Knowledge Expansion</a></div>
    <div class="paper-meta">
      📅 2025-02-18
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to new and diverse knowledge is essential for their lasting effectiveness in real-world applications. This survey provides an overview of state-of-the-art methods for expanding the knowledge of LLMs, focusing on integrating various knowledge types, including factual information, domain expertise, language proficiency, and user preferences. We explore techniques, such as continual learning, model editing, and retrieval-based explicit adaptation, while discussing challenges like knowledge consistency and scalability. Designed as a guide for researchers and practitioners, this survey sheds light on opportunities for advancing LLMs as adaptable and robust knowledge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09077v3">CuriousLLM: Elevating Multi-Document Question Answering with LLM-Enhanced Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-18
      | 💬 Accepted for publication in NAACL 2025. The official version will be available in the ACL Anthology
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in open-domain question answering. However, they continue to face challenges such as hallucinations and knowledge cutoffs. These issues can be mitigated through in-context learning by providing LLMs with relevant context before generating answers. Recent literature proposes Knowledge Graph Prompting (KGP) which integrates knowledge graphs with an LLM-based traversal agent to substantially enhance document retrieval quality. However, KGP requires costly fine-tuning with large datasets and remains prone to hallucination. In this paper, we propose CuriousLLM, an enhancement that integrates a curiosity-driven reasoning mechanism into an LLM agent. This mechanism enables the agent to generate relevant follow-up questions, thereby guiding the information retrieval process more efficiently. Central to our approach is the development of the new Follow-upQA dataset, which includes questions and supporting evidence as input, with follow-up questions serving as ground truths. These follow-up questions either inquire about what is still missing to fully answer the user's query or use special tokens to signify that the retrieved evidence is sufficient. Our experiments show that CuriousLLM significantly boosts LLM performance in multi-document question answering (MD-QA), circumventing the substantial computational costs and latency from the original KGP framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11578v1">Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in natural language generation but often face challenges in tasks requiring precise calculations and structural analysis. This paper investigates the performance of state-of-the-art LLMs on language complexity measurement tasks, through the computation of the LIX readability metric and Average Dependency Distance (ADD). Using Swedish high school and university-level essays, we evaluate the models' abilities to compute LIX scores and perform dependency parsing, comparing their results to established ground truths. Our findings reveal that while all models demonstrate some capacity for these tasks, ChatGPT-o1-mini performs most consistently, achieving the highest accuracy in both LIX computation and dependency parsing. Additionally, we observe a strong significant correlation -0.875 p 0.026 (N=6) between the models' accuracy in computing LIX and their overall performance on the Massive Multitask Language Understanding (MMLU) benchmark. These results suggest that language complexity measurement abilities can serve as a noisy zero-shot proxies for assessing the general capabilities of LLMs, providing a practical method for model evaluation without the need for extensive benchmarking datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16205v4">LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23166v2">SciPIP: An LLM-based Scientific Paper Idea Proposer</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 20 pages, 5 figures, 12 tables. The code has been availabel: https://github.com/cheerss/SciPIP
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for automating the proposal of innovative scientific ideas. This process involves two key phases: literature retrieval and idea generation. However, existing approaches often fall short due to their reliance on keyword-based search tools during the retrieval phase, which neglects crucial semantic information and frequently results in incomplete retrieval outcomes. Similarly, in the idea generation phase, current methodologies tend to depend solely on the internal knowledge of LLMs or metadata from retrieved papers, thereby overlooking significant valuable insights contained within the full texts. To address these limitations, we introduce SciPIP, an innovative framework designed to enhance the LLM-based proposal of scientific ideas through improvements in both literature retrieval and idea generation. Our approach begins with the construction of a comprehensive literature database that supports advanced retrieval based not only on keywords but also on semantics and citation relationships. This is complemented by the introduction of a multi-granularity retrieval algorithm aimed at ensuring more thorough and exhaustive retrieval results. For the idea generation phase, we propose a dual-path framework that effectively integrates both the content of retrieved papers and the extensive internal knowledge of LLMs. This integration significantly boosts the novelty, feasibility, and practical value of proposed ideas. Our experiments, conducted across various domains such as natural language processing and computer vision, demonstrate SciPIP's capability to generate a multitude of innovative and useful ideas. These findings underscore SciPIP's potential as a valuable tool for researchers seeking to advance their fields with groundbreaking concepts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22662v2">EMOS: Embodiment-aware Heterogeneous Multi-robot Operating System with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 10 pages of main content, 3 pages of references, 5 pages of appendix, 7 figures in total
    </div>
    <details class="paper-abstract">
      Heterogeneous multi-robot systems (HMRS) have emerged as a powerful approach for tackling complex tasks that single robots cannot manage alone. Current large-language-model-based multi-agent systems (LLM-based MAS) have shown success in areas like software development and operating systems, but applying these systems to robot control presents unique challenges. In particular, the capabilities of each agent in a multi-robot system are inherently tied to the physical composition of the robots, rather than predefined roles. To address this issue, we introduce a novel multi-agent framework designed to enable effective collaboration among heterogeneous robots with varying embodiments and capabilities, along with a new benchmark named Habitat-MAS. One of our key designs is $\textit{Robot Resume}$: Instead of adopting human-designed role play, we propose a self-prompted approach, where agents comprehend robot URDF files and call robot kinematics tools to generate descriptions of their physics capabilities to guide their behavior in task planning and action execution. The Habitat-MAS benchmark is designed to assess how a multi-agent framework handles tasks that require embodiment-aware reasoning, which includes 1) manipulation, 2) perception, 3) navigation, and 4) comprehensive multi-floor object rearrangement. The experimental results indicate that the robot's resume and the hierarchical design of our multi-agent system are essential for the effective operation of the heterogeneous multi-robot system within this intricate problem context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11544v1">Evaluating o1-Like LLMs: Unlocking Reasoning for Translation through Comprehensive Analysis</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The o1-Like LLMs are transforming AI by simulating human cognitive processes, but their performance in multilingual machine translation (MMT) remains underexplored. This study examines: (1) how o1-Like LLMs perform in MMT tasks and (2) what factors influence their translation quality. We evaluate multiple o1-Like LLMs and compare them with traditional models like ChatGPT and GPT-4o. Results show that o1-Like LLMs establish new multilingual translation benchmarks, with DeepSeek-R1 surpassing GPT-4o in contextless tasks. They demonstrate strengths in historical and cultural translation but exhibit a tendency for rambling issues in Chinese-centric outputs. Further analysis reveals three key insights: (1) High inference costs and slower processing speeds make complex translation tasks more resource-intensive. (2) Translation quality improves with model size, enhancing commonsense reasoning and cultural translation. (3) The temperature parameter significantly impacts output quality-lower temperatures yield more stable and accurate translations, while higher temperatures reduce coherence and precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03138v2">Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated impressive performance in molecular generation, which offers potential to accelerate drug discovery. However, the current LLMs overlook a critical requirement for drug discovery: proposing a diverse set of molecules. This diversity is essential for improving the chances of finding a viable drug, as it provides alternative molecules that may succeed where others fail in real-world validations. Nevertheless, the LLMs often output structurally similar molecules. While decoding schemes like diverse beam search may enhance textual diversity, this often does not align with molecular structural diversity. In response, we propose a new method for fine-tuning molecular generative LLMs to autoregressively generate a set of structurally diverse molecules, where each molecule is generated by conditioning on the previously generated molecules. Our approach consists of two stages: (1) supervised fine-tuning to adapt LLMs to autoregressively generate molecules in a sequence and (2) reinforcement learning to maximize structural diversity within the generated molecules. Our experiments show that the proposed approach enables LLMs to generate diverse molecules better than existing approaches for diverse sequence generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11686v4">CCoE: A Compact and Efficient LLM Framework with Multi-Expert Collaboration for Resource-Limited Settings</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved exceptional performance across diverse domains through training on massive datasets. However, scaling LLMs to support multiple downstream domain applications remains a significant challenge, especially under resource constraints. Existing approaches often struggle to balance performance across multiple domains with resource efficiency, limiting their broader applicability. To address this, we introduce the CCoE architecture, a modular framework that seamlessly integrates domain-specific experts into a unified LLM. By leveraging independently trained expert subnetworks on a shared backbone partition, CCoE achieves state-of-the-art performance while significantly reducing the resource requirements for multi-expert deployments. Furthermore, rule-based gating and expert planning in CCoE enable flexible task allocation, promoting expert collaboration to handle complex reasoning tasks. CCoE not only reduces inference costs but also provides a flexible and scalable solution for integrating domain expertise across diverse applications. Experiments on five domains demonstrate that CCoE achieves comparable performance to current domain-specific LLMs. Moreover, compared to existing multi-domain model ensemble methods, CCoE reduces memory usage by 61.3%, while improving inference efficiency by 0.76x over parameter-efficient multi-expert integration approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11533v1">Be Cautious When Merging Unfamiliar LLMs: A Phishing Model Capable of Stealing Privacy</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Model merging is a widespread technology in large language models (LLMs) that integrates multiple task-specific LLMs into a unified one, enabling the merged model to inherit the specialized capabilities of these LLMs. Most task-specific LLMs are sourced from open-source communities and have not undergone rigorous auditing, potentially imposing risks in model merging. This paper highlights an overlooked privacy risk: \textit{an unsafe model could compromise the privacy of other LLMs involved in the model merging.} Specifically, we propose PhiMM, a privacy attack approach that trains a phishing model capable of stealing privacy using a crafted privacy phishing instruction dataset. Furthermore, we introduce a novel model cloaking method that mimics a specialized capability to conceal attack intent, luring users into merging the phishing model. Once victims merge the phishing model, the attacker can extract personally identifiable information (PII) or infer membership information (MI) by querying the merged model with the phishing instruction. Experimental results show that merging a phishing model increases the risk of privacy breaches. Compared to the results before merging, PII leakage increased by 3.9\% and MI leakage increased by 17.4\% on average. We release the code of PhiMM through a link.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11521v1">DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years. In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12480v2">KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 under reveiw; new results and analysis added, typos corrected
    </div>
    <details class="paper-abstract">
      Schema matching (SM) and entity matching (EM) tasks are crucial for data integration. While large language models (LLMs) have shown promising results in these tasks, they suffer from hallucinations and confusion about task instructions. This study presents the Knowledge-Compliant Matching Framework (KcMF), an LLM-based approach that addresses these issues without the need for domain-specific fine-tuning. KcMF employs a once-and-for-all pseudo-code-based task decomposition strategy to adopt natural language statements that guide LLM reasoning and reduce confusion across various task types. We also propose two mechanisms, Dataset as Knowledge (DaK) and Example as Knowledge (EaK), to build domain knowledge sets when unstructured domain knowledge is lacking. Moreover, we introduce a result-ensemble strategy to leverage multiple knowledge sources and suppress badly formatted outputs. Extensive evaluations confirm that KcMF clearly enhances five LLM backbones in both SM and EM tasks while outperforming the non-LLM competitors by an average F1-score of 17.93%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v4">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted by ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16756v3">How Well Do LLMs Handle Cantonese? Benchmarking Cantonese Capabilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has transformed the competitive landscape in natural language processing (NLP), particularly for English and other data-rich languages. However, underrepresented languages like Cantonese, spoken by over 85 million people, face significant development gaps, which is particularly concerning given the economic significance of the Guangdong-Hong Kong-Macau Greater Bay Area, and in substantial Cantonese-speaking populations in places like Singapore and North America. Despite its wide use, Cantonese has scant representation in NLP research, especially compared to other languages from similarly developed regions. To bridge these gaps, we outline current Cantonese NLP methods and introduce new benchmarks designed to evaluate LLM performance in factual generation, mathematical logic, complex reasoning, and general knowledge in Cantonese, which aim to advance open-source Cantonese LLM technology. We also propose future research directions and recommended models to enhance Cantonese LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11493v1">DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face computational inefficiencies and redundant processing when handling long context inputs, prompting a focus on compression techniques. While existing semantic vector-based compression methods achieve promising performance, these methods fail to account for the intrinsic information density variations between context chunks, instead allocating soft tokens uniformly across context chunks. This uniform distribution inevitably diminishes allocation to information-critical regions. To address this, we propose Dynamic Allocation of Soft Tokens (DAST), a simple yet effective method that leverages the LLM's intrinsic understanding of contextual relevance to guide compression. DAST combines perplexity-based local information with attention-driven global information to dynamically allocate soft tokens to the informative-rich chunks, enabling effective, context-aware compression. Experimental results across multiple benchmarks demonstrate that DAST surpasses state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15549v2">WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, aligning these models with human preferences has emerged as a critical challenge. Traditional alignment methods, relying on human or LLM annotated datasets, are limited by their resource-intensive nature, inherent subjectivity, misalignment with real-world user preferences, and the risk of feedback loops that amplify model biases. To overcome these limitations, we introduce WildFeedback, a novel framework that leverages in-situ user feedback during conversations with LLMs to create preference datasets automatically. Given a corpus of multi-turn user-LLM conversation, WildFeedback identifies and classifies user feedback to LLM responses between conversation turns. The user feedback is then used to create examples of preferred and dispreferred responses according to users' preference. Our experiments demonstrate that LLMs fine-tuned on WildFeedback dataset exhibit significantly improved alignment with user preferences, as evidenced by both traditional benchmarks and our proposed checklist-guided evaluation. By incorporating in-situ feedback from actual users, WildFeedback addresses the scalability, subjectivity, and bias challenges that plague existing approaches, marking a significant step toward developing LLMs that are more responsive to the diverse and evolving needs of their users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11471v1">GLTW: Joint Improved Graph Transformer and LLM via Three-Word Language for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Knowledge Graph Completion (KGC), which aims to infer missing or incomplete facts, is a crucial task for KGs. However, integrating the vital structural information of KGs into Large Language Models (LLMs) and outputting predictions deterministically remains challenging. To address this, we propose a new method called GLTW, which encodes the structural information of KGs and merges it with LLMs to enhance KGC performance. Specifically, we introduce an improved Graph Transformer (iGT) that effectively encodes subgraphs with both local and global structural information and inherits the characteristics of language model, bypassing training from scratch. Also, we develop a subgraph-based multi-classification training objective, using all entities within KG as classification objects, to boost learning efficiency.Importantly, we combine iGT with an LLM that takes KG language prompts as input.Our extensive experiments on various KG datasets show that GLTW achieves significant performance gains compared to SOTA baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09077v2">CuriousLLM: Elevating Multi-Document Question Answering with LLM-Enhanced Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved significant success in open-domain question answering. However, they continue to face challenges such as hallucinations and knowledge cutoffs. These issues can be mitigated through in-context learning by providing LLMs with relevant context before generating answers. Recent literature proposes Knowledge Graph Prompting (KGP) which integrates knowledge graphs with an LLM-based traversal agent to substantially enhance document retrieval quality. However, KGP requires costly fine-tuning with large datasets and remains prone to hallucination. In this paper, we propose CuriousLLM, an enhancement that integrates a curiosity-driven reasoning mechanism into an LLM agent. This mechanism enables the agent to generate relevant follow-up questions, thereby guiding the information retrieval process more efficiently. Central to our approach is the development of the new Follow-upQA dataset, which includes questions and supporting evidence as input, with follow-up questions serving as ground truths. These follow-up questions either inquire about what is still missing to fully answer the user's query or use special tokens to signify that the retrieved evidence is sufficient. Our experiments show that CuriousLLM significantly boosts LLM performance in multi-document question answering (MD-QA), circumventing the substantial computational costs and latency from the original KGP framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11451v1">From Personas to Talks: Revisiting the Impact of Personas on LLM-Synthesized Emotional Support Conversations</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized the generation of emotional support conversations (ESC), offering scalable solutions with reduced costs and enhanced data privacy. This paper explores the role of personas in the creation of ESC by LLMs. Our research utilizes established psychological frameworks to measure and infuse persona traits into LLMs, which then generate dialogues in the emotional support scenario. We conduct extensive evaluations to understand the stability of persona traits in dialogues, examining shifts in traits post-generation and their impact on dialogue quality and strategy distribution. Experimental results reveal several notable findings: 1) LLMs can infer core persona traits, 2) subtle shifts in emotionality and extraversion occur, influencing the dialogue dynamics, and 3) the application of persona traits modifies the distribution of emotional support strategies, enhancing the relevance and empathetic quality of the responses. These findings highlight the potential of persona-driven LLMs in crafting more personalized, empathetic, and effective emotional support dialogues, which has significant implications for the future design of AI-driven emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11441v1">Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) risk retaining unauthorized or sensitive information from their training data, which raises privacy concerns. LLM unlearning seeks to mitigate these risks by selectively removing specified data while maintaining overall model performance. However, most existing work focus on methods to achieve effective forgetting and does not provide a detailed analysis of the retain set, the portion of training data that is not targeted for removal. In this paper, we investigate the effects of unlearning on various subsets of the retain set through a case study on entity unlearning. We introduce the Syntactically Similar Neighbor Set, a group of queries that share similar syntactic structures with the data targeted for removal, and show that this subset suffers the greatest performance drop during unlearning. Moreover, when used for regularization, this set not only preserves performance on syntactically similar queries but also delivers comparable or improved results across other data subsets. Our results highlight that syntactic similarity is a critical factor, potentially more so than domain or entity relationships, in achieving effective and practical LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11436v1">ADO: Automatic Data Optimization for Inputs in LLM Prompts</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      This study explores a novel approach to enhance the performance of Large Language Models (LLMs) through the optimization of input data within prompts. While previous research has primarily focused on refining instruction components and augmenting input data with in-context examples, our work investigates the potential benefits of optimizing the input data itself. We introduce a two-pronged strategy for input data optimization: content engineering and structural reformulation. Content engineering involves imputing missing values, removing irrelevant attributes, and enriching profiles by generating additional information inferred from existing attributes. Subsequent to content engineering, structural reformulation is applied to optimize the presentation of the modified content to LLMs, given their sensitivity to input format. Our findings suggest that these optimizations can significantly improve the performance of LLMs in various tasks, offering a promising avenue for future research in prompt engineering. The source code is available at https://anonymous.4open.science/r/ADO-6BC5/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11433v1">\textsc{FLAG-Trader}: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05551v2">FRAMES: Boosting LLMs with A Four-Quadrant Multi-Stage Pretraining Strategy</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced human language understanding and generation, with pretraining data quality and organization being crucial to their performance. Multi-stage pretraining is a promising approach, but existing methods often lack quantitative criteria for data partitioning and instead rely on intuitive heuristics. In this paper, we propose the novel Four-quadRAnt Multi-stage prEtraining strategy (FRAME), guided by the established principle of organizing the pretraining process into four stages to achieve significant loss reductions four times. This principle is grounded in two key findings: first, training on high Perplexity (PPL) data followed by low PPL data, and second, training on low PPL difference (PD) data followed by high PD data, both causing the loss to drop significantly twice and performance enhancements. By partitioning data into four quadrants and strategically organizing them, FRAME achieves a remarkable 16.8% average improvement over random across MMLU and CMMLU for the 3B model, effectively boosting LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14141v2">LLMs can Realize Combinatorial Creativity: Generating Creative Ideas via LLMs for Scientific Research</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Scientific idea generation has been extensively studied in creativity theory and computational creativity research, providing valuable frameworks for understanding and implementing creative processes. However, recent work using Large Language Models (LLMs) for research idea generation often overlooks these theoretical foundations. We present a framework that explicitly implements combinatorial creativity theory using LLMs, featuring a generalization-level retrieval system for cross-domain knowledge discovery and a structured combinatorial process for idea generation. The retrieval system maps concepts across different abstraction levels to enable meaningful connections between disparate domains, while the combinatorial process systematically analyzes and recombines components to generate novel solutions. Experiments on the OAG-Bench dataset demonstrate our framework's effectiveness, consistently outperforming baseline approaches in generating ideas that align with real research developments (improving similarity scores by 7\%-10\% across multiple metrics). Our results provide strong evidence that LLMs can effectively realize combinatorial creativity when guided by appropriate theoretical frameworks, contributing both to practical advancement of AI-assisted research and theoretical understanding of machine creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11417v1">DiSCo: Device-Server Collaborative LLM-Based Text Streaming Services</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 17 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) in text streaming services has introduced significant cost and Quality of Experience (QoE) challenges in serving millions of daily requests, especially in meeting Time-To-First-Token (TTFT) and Time-Between-Token (TBT) requirements for real-time interactions. Our real-world measurements show that both server-based and on-device deployments struggle to meet diverse QoE demands: server deployments face high costs and last-hop issues (e.g., Internet latency and dynamics), while on-device LLM inference is constrained by resources. We introduce DiSCo, a device-server cooperative scheduler designed to optimize users' QoE by adaptively routing requests and migrating response generation between endpoints while maintaining cost constraints. DiSCo employs cost-aware scheduling, leveraging the predictable speed of on-device LLM inference with the flexible capacity of server-based inference to dispatch requests on the fly, while introducing a token-level migration mechanism to ensure consistent token delivery during migration. Evaluations on real-world workloads -- including commercial services like OpenAI GPT and DeepSeek, and open-source deployments such as LLaMA3 -- show that DiSCo can improve users' QoE by reducing tail TTFT (11-52\%) and mean TTFT (6-78\%) across different model-device configurations, while dramatically reducing serving costs by up to 84\% through its migration mechanism while maintaining comparable QoE levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13949v2">Mufu: Multilingual Fused Learning for Low-Resource Translation with LLM</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) are great translators, but this is largely limited to high-resource languages. For many LLMs, translating in and out of low-resource languages remains a challenging task. To maximize data efficiency in this low-resource setting, we introduce Mufu, which includes a selection of automatically generated multilingual candidates and an instruction to correct inaccurate translations in the prompt. Mufu prompts turn a translation task into a postediting one, and seek to harness the LLM's reasoning capability with auxiliary translation candidates, from which the model is required to assess the input quality, align the semantics cross-lingually, copy from relevant inputs and override instances that are incorrect. Our experiments on En-XX translations over the Flores-200 dataset show LLMs finetuned against Mufu-style prompts are robust to poor quality auxiliary translation candidates, achieving performance superior to NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs. We then distill these models to reduce inference cost, while maintaining on average 3.1 chrF improvement over finetune-only baseline in low-resource translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07191v4">Bag of Tricks for Inference-time Computation of LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      With the advancement of large language models (LLMs), solving complex reasoning tasks has gained increasing attention. Inference-time computation methods (e.g., Best-of-N, beam search, et al.) are particularly valuable as they can enhance reasoning performance without modifying model parameters or requiring additional training. However, these techniques come with implementation challenges, and most existing methods remain at the proof-of-concept stage with limited practical adoption due to their computational complexity and varying effectiveness across different tasks. In this paper, we investigate and benchmark diverse inference-time computation strategies across reasoning tasks of varying complexity. Since most current methods rely on a proposer-verifier pipeline that first generates candidate solutions (e.g., reasoning solutions) and then selects the best one based on reward signals (e.g., RLHF rewards, process rewards), our research focuses on optimizing both candidate solution generation (e.g., instructing prompts, hyperparameters such as temperature and top-p) and reward mechanisms (e.g., self-evaluation, reward types). Through extensive experiments (more than 20,000 A100-80G GPU hours with over 1,000 experiments) across a variety of models (e.g., Llama, Qwen, and Mistral families) of various sizes, our ablation studies reveal that previously overlooked strategies can significantly enhance performance (e.g., tuning temperature can improve reasoning task performance by up to 5%). Furthermore, we establish a standardized benchmark for inference-time computation by systematically evaluating six representative methods across eight reasoning tasks. These findings provide a stronger foundation for future research. The code is available at https://github.com/usail-hkust/benchmark_inference_time_computation_LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02795v3">InfiFusion: A Unified Framework for Enhanced Cross-Model Reasoning via LLM Fusion</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Significant performance improvements over the previous version; under review;
    </div>
    <details class="paper-abstract">
      We introduce InfiFusion, an efficient training pipeline designed to integrate multiple domain-specialized Large Language Models (LLMs) into a single pivot model, effectively harnessing the strengths of each source model. Traditional fusion methods either merge model parameters directly or rely on knowledge distillation with rigid assumptions, limiting their flexibility and efficiency. InfiFusion overcomes these limitations by enhancing Universal Logit Distillation (ULD) with Top-K selection and Logits Standardization. We propose two fusion strategies: Pairwise Fusion (InfiFusion$_p$), where each source model knowledge is distilled individually into the pivot model followed by merging and Unified Fusion (InfiFusion$_u$), where knowledge from all source models is distilled simultaneously into the pivot model. InfiFusion outperforms the state-of-the-art models, such as Qwen-2.5-14B-Instruct and Phi-4, across 11 widely applied benchmarks covering reasoning, coding, mathematics, and instruction-following tasks. Notably, InfiFusion achieves this superior performance while significantly reduces computational costs, completing full training with only 160 H800 GPU hours compared to the millions typically required for traditional LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09412v2">FB-Bench: A Fine-Grained Multi-Task Benchmark for Evaluating LLMs' Responsiveness to Human Feedback</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Human feedback is crucial in the interactions between humans and Large Language Models (LLMs). However, existing research primarily focuses on benchmarking LLMs in single-turn dialogues. Even in benchmarks designed for multi-turn dialogues, the user inputs are often independent, neglecting the nuanced and complex nature of human feedback within real-world usage scenarios. To fill this research gap, we introduce FB-Bench, a fine-grained, multi-task benchmark designed to evaluate LLMs' responsiveness to human feedback under real-world usage scenarios in Chinese. Drawing from the two main interaction scenarios, FB-Bench comprises 591 meticulously curated samples, encompassing eight task types, five deficiency types of response, and nine feedback types. We extensively evaluate a broad array of popular LLMs, revealing significant variations in their performance across different interaction scenarios. Further analysis indicates that task, human feedback, and deficiencies of previous responses can also significantly impact LLMs' responsiveness. Our findings underscore both the strengths and limitations of current models, providing valuable insights and directions for future research. Code and datasets are available at https://github.com/PKU-Baichuan-MLSystemLab/FB-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11401v1">Following the Autoregressive Nature of LLM Embeddings via Compression and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      A new trend uses LLMs as dense text encoders via contrastive learning. However, since LLM embeddings predict the probability distribution of the next token, they are inherently generative and distributive, conflicting with contrastive learning, which requires embeddings to capture full-text semantics and align via cosine similarity. This discrepancy hinders the full utilization of LLMs' pre-training capabilities, resulting in inefficient learning. In response to this issue, we propose AutoRegEmbed, a new contrastive learning method built on embedding conditional probability distributions, which integrates two core tasks: information compression and conditional distribution alignment. The information compression task encodes text into the embedding space, ensuring that the embedding vectors capture global semantics. The conditional distribution alignment task focuses on aligning text embeddings with positive samples embeddings by leveraging the conditional distribution of embeddings while simultaneously reducing the likelihood of generating negative samples from text embeddings, thereby achieving embedding alignment and uniformity. Experimental results demonstrate that our method significantly outperforms traditional contrastive learning approaches and achieves performance comparable to state-of-the-art models when using the same amount of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11400v1">Revisiting Robust RAG: Do We Still Need Complex Robust Training in the Era of Powerful LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) systems often suffer from performance degradation when encountering noisy or irrelevant documents, driving researchers to develop sophisticated training strategies to enhance their robustness against such retrieval noise. However, as large language models (LLMs) continue to advance, the necessity of these complex training methods is increasingly questioned. In this paper, we systematically investigate whether complex robust training strategies remain necessary as model capacity grows. Through comprehensive experiments spanning multiple model architectures and parameter scales, we evaluate various document selection methods and adversarial training techniques across diverse datasets. Our extensive experiments consistently demonstrate that as models become more powerful, the performance gains brought by complex robust training methods drop off dramatically. We delve into the rationale and find that more powerful models inherently exhibit superior confidence calibration, better generalization across datasets (even when trained with randomly selected documents), and optimal attention mechanisms learned with simpler strategies. Our findings suggest that RAG systems can benefit from simpler architectures and training strategies as models become more powerful, enabling more scalable applications with minimal complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12445v2">Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Evaluation for Korean LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      The Open Ko-LLM Leaderboard has been instrumental in benchmarking Korean Large Language Models (LLMs), yet it has certain limitations. Notably, the disconnect between quantitative improvements on the overly academic leaderboard benchmarks and the qualitative impact of the models should be addressed. Furthermore, the benchmark suite is largely composed of translated versions of their English counterparts, which may not fully capture the intricacies of the Korean language. To address these issues, we propose Open Ko-LLM Leaderboard2, an improved version of the earlier Open Ko-LLM Leaderboard. The original benchmarks are entirely replaced with new tasks that are more closely aligned with real-world capabilities. Additionally, four new native Korean benchmarks are introduced to better reflect the distinct characteristics of the Korean language. Through these refinements, Open Ko-LLM Leaderboard2 seeks to provide a more meaningful evaluation for advancing Korean LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11393v1">HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in commonsense reasoning; however, some variations in questions can trigger incorrect responses. Do these models truly understand commonsense knowledge, or just memorize expression patterns? To investigate this question, we present the first extensive robustness evaluation of LLMs in commonsense reasoning. We introduce HellaSwag-Pro, a large-scale bilingual benchmark consisting of 11,200 cases, by designing and compiling seven types of question variants. To construct this benchmark, we propose a two-stage method to develop Chinese HellaSwag, a finely annotated dataset comprising 12,000 instances across 56 categories. We conduct extensive experiments on 41 representative LLMs, revealing that these LLMs are far from robust in commonsense reasoning. Furthermore, this robustness varies depending on the language in which the LLM is tested. This work establishes a high-quality evaluation benchmark, with extensive experiments offering valuable insights to the community in commonsense reasoning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00804v2">Examining Identity Drift in Conversations of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show impressive conversational abilities but sometimes show identity drift problems, where their interaction patterns or styles change over time. As the problem has not been thoroughly examined yet, this study examines identity consistency across nine LLMs. Specifically, we (1) investigate whether LLMs could maintain consistent patterns (or identity) and (2) analyze the effect of the model family, parameter sizes, and provided persona types. Our experiments involve multi-turn conversations on personal themes, analyzed in qualitative and quantitative ways. Experimental results indicate three findings. (1) Larger models experience greater identity drift. (2) Model differences exist, but their effect is not stronger than parameter sizes. (3) Assigning a persona may not help to maintain identity. We hope these three findings can help to improve persona stability in AI-driven dialogue systems, particularly in long-term conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11380v1">Exploring the Small World of Word Embeddings: A Comparative Study on Conceptual Spaces from LLMs of Different Scales</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Paper under review
    </div>
    <details class="paper-abstract">
      A conceptual space represents concepts as nodes and semantic relatedness as edges. Word embeddings, combined with a similarity metric, provide an effective approach to constructing such a space. Typically, embeddings are derived from traditional distributed models or encoder-only pretrained models, whose objectives directly capture the meaning of the current token. In contrast, decoder-only models, including large language models (LLMs), predict the next token, making their embeddings less directly tied to the current token's semantics. Moreover, comparative studies on LLMs of different scales remain underexplored. In this paper, we construct a conceptual space using word embeddings from LLMs of varying scales and comparatively analyze their properties. We establish a network based on a linguistic typology-inspired connectivity hypothesis, examine global statistical properties, and compare LLMs of varying scales. Locally, we analyze conceptual pairs, WordNet relations, and a cross-lingual semantic network for qualitative words. Our results indicate that the constructed space exhibits small-world properties, characterized by a high clustering coefficient and short path lengths. Larger LLMs generate more intricate spaces, with longer paths reflecting richer relational structures and connections. Furthermore, the network serves as an efficient bridge for cross-lingual semantic mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01558v2">Predicting the Performance of Black-box LLMs through Self-Queries</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly relied on in AI systems, predicting when they make mistakes is crucial. While a great deal of work in the field uses internal representations to interpret model behavior, these representations are inaccessible when given solely black-box access through an API. In this paper, we extract features of LLMs in a black-box manner by using follow-up prompts and taking the probabilities of different responses as representations to train reliable predictors of model behavior. We demonstrate that training a linear model on these low-dimensional representations produces reliable and generalizable predictors of model performance at the instance level (e.g., if a particular generation correctly answers a question). Remarkably, these can often outperform white-box linear predictors that operate over a model's hidden state or the full distribution over its vocabulary. In addition, we demonstrate that these extracted features can be used to evaluate more nuanced aspects of a language model's state. For instance, they can be used to distinguish between a clean version of GPT-4o-mini and a version that has been influenced via an adversarial system prompt that answers question-answering tasks incorrectly or introduces bugs into generated code. Furthermore, they can reliably distinguish between different model architectures and sizes, enabling the detection of misrepresented models provided through an API (e.g., identifying if GPT-3.5 is supplied instead of GPT-4o-mini).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11368v1">LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 26 pages, 6 figures, 15 tables
    </div>
    <details class="paper-abstract">
      The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus for reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13276v4">SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Attention is the cornerstone of modern Large Language Models (LLMs). Yet its quadratic complexity hinders efficiency and scalability, especially for long-context processing. A promising approach is to leverage sparsity in attention. However, existing sparsity-based solutions predominantly rely on predefined patterns or heuristics at the attention head level, struggling to adapt dynamically to different contexts efficiently. We propose SeerAttention, a simple yet effective attention mechanism that directly learns the block-level attention sparsity from the LLM itself. Inspired by the gating mechanism in Mixture of Experts (MoE), SeerAttention augments the conventional attention with a learnable gate that selectively activates important blocks within the attention map. Specifically, the gate first pools the query (Q) and key (K) tensors along the sequence dimension and processes them through learnable linear layers. The resulting matrices are then multiplied together to produce the gating scores, which are used to predict block-level attention sparsity. Combined with our block-sparse FlashAttention kernel, SeerAttention can achieve significant speedup on GPUs. When applied to pre-trained LLMs, SeerAttention only requires training the gate parameters in a lightweight self-distillation manner, allowing rapid convergence. Our evaluation results demonstrate that SeerAttention achieves better model accuracy and lower latency for long-context pre-filling compared to prior methods. Code is available at: https://github.com/microsoft/SeerAttention
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13115v2">Dagger Behind Smile: Fool LLMs with a Happy Ending Story</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      The wide adoption of Large Language Models (LLMs) has attracted significant attention from $\textit{jailbreak}$ attacks, where adversarial prompts crafted through optimization or manual design exploit LLMs to generate malicious contents. However, optimization-based attacks have limited efficiency and transferability, while existing manual designs are either easily detectable or demand intricate interactions with LLMs. In this paper, we first point out a novel perspective for jailbreak attacks: LLMs are more responsive to $\textit{positive}$ prompts. Based on this, we deploy Happy Ending Attack (HEA) to wrap up a malicious request in a scenario template involving a positive prompt formed mainly via a $\textit{happy ending}$, it thus fools LLMs into jailbreaking either immediately or at a follow-up malicious request.This has made HEA both efficient and effective, as it requires only up to two turns to fully jailbreak LLMs. Extensive experiments show that our HEA can successfully jailbreak on state-of-the-art LLMs, including GPT-4o, Llama3-70b, Gemini-pro, and achieves 88.79\% attack success rate on average. We also provide quantitative explanations for the success of HEA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11358v1">Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 15 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11355v1">"Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Our code will be available at https://github.com/pillowsofwind/LLM-CBRN-Risks
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03257v2">Understanding LLM Development Through Longitudinal Study: Insights from the Open Ko-LLM Leaderboard</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 Accepted to NAACL 2025 Industry
    </div>
    <details class="paper-abstract">
      This paper conducts a longitudinal study over eleven months to address the limitations of prior research on the Open Ko-LLM Leaderboard, which have relied on empirical studies with restricted observation periods of only five months. By extending the analysis duration, we aim to provide a more comprehensive understanding of the progression in developing Korean large language models (LLMs). Our study is guided by three primary research questions: (1) What are the specific challenges in improving LLM performance across diverse tasks on the Open Ko-LLM Leaderboard over time? (2) How does model size impact task performance correlations across various benchmarks? (3) How have the patterns in leaderboard rankings shifted over time on the Open Ko-LLM Leaderboard?. By analyzing 1,769 models over this period, our research offers a comprehensive examination of the ongoing advancements in LLMs and the evolving nature of evaluation frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12372v1">Factual Inconsistency in Data-to-Text Generation Scales Exponentially with LLM Size: A Statistical Validation</a></div>
    <div class="paper-meta">
      📅 2025-02-17
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Monitoring factual inconsistency is essential for ensuring trustworthiness in data-to-text generation (D2T). While large language models (LLMs) have demonstrated exceptional performance across various D2T tasks, previous studies on scaling laws have primarily focused on generalization error through power law scaling to LLM size (i.e., the number of model parameters). However, no research has examined the impact of LLM size on factual inconsistency in D2T. In this paper, we investigate how factual inconsistency in D2T scales with LLM size by exploring two scaling laws: power law and exponential scaling. To rigorously evaluate and compare these scaling laws, we employ a statistical validation framework consisting of three key stages: predictive performance estimation, goodness-of-fit assessment, and comparative analysis. For a comprehensive empirical study, we analyze three popular LLM families across five D2T datasets, measuring factual inconsistency inversely using four state-of-the-art consistency metrics. Our findings, based on exhaustive empirical results and validated through our framework, reveal that, contrary to the widely assumed power law scaling, factual inconsistency in D2T follows an exponential scaling with LLM size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12340v1">Understanding Silent Data Corruption in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      As the scale of training large language models (LLMs) increases, one emergent failure is silent data corruption (SDC), where hardware produces incorrect computations without explicit failure signals. In this work, we are the first to investigate the impact of real-world SDCs on LLM training by comparing model training between healthy production nodes and unhealthy nodes exhibiting SDCs. With the help from a cloud computing platform, we access the unhealthy nodes that were swept out from production by automated fleet management. Using deterministic execution via XLA compiler and our proposed synchronization mechanisms, we isolate and analyze the impact of SDC errors on these nodes at three levels: at each submodule computation, at a single optimizer step, and at a training period. Our results reveal that the impact of SDCs on computation varies on different unhealthy nodes. Although in most cases the perturbations from SDCs on submodule computation and gradients are relatively small, SDCs can lead models to converge to different optima with different weights and even cause spikes in the training loss. Our analysis sheds light on further understanding and mitigating the impact of SDCs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12325v1">From Dense to Dynamic: Token-Difficulty Driven MoEfication of Pre-Trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) for different inference constraints is computationally expensive, limiting control over efficiency-accuracy trade-offs. Moreover, once trained, these models typically process tokens uniformly, regardless of their complexity, leading to static and inflexible behavior. In this paper, we introduce a post-training optimization framework, DynaMoE, that adapts a pre-trained dense LLM to a token-difficulty-driven Mixture-of-Experts model with minimal fine-tuning cost. This adaptation makes the model dynamic, with sensitivity control to customize the balance between efficiency and accuracy. DynaMoE features a token-difficulty-aware router that predicts the difficulty of tokens and directs them to the appropriate sub-networks or experts, enabling larger experts to handle more complex tokens and smaller experts to process simpler ones. Our experiments demonstrate that DynaMoE can generate a range of adaptive model variants of the existing trained LLM with a single fine-tuning step, utilizing only $10B$ tokens, a minimal cost compared to the base model's training. Each variant offers distinct trade-offs between accuracy and performance. Compared to the baseline post-training optimization framework, Flextron, our method achieves similar aggregated accuracy across downstream tasks, despite using only $\frac{1}{9}\text{th}$ of their fine-tuning cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12275v1">Integrating Expert Knowledge into Logical Programs via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma, Mixtral, Mistral, and Qwen. Results reveal that while models generate nearly perfect syntactically correct code, they frequently exhibit logical errors in translating expert knowledge. Furthermore, iterative self-correction yields only marginal improvements (up to 3%). Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered. The complete implementation, along with all relevant data, is available at GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09730v2">Sociodemographic Prompting is Not Yet an Effective Approach for Simulating Subjective Judgments with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Human judgments are inherently subjective and are actively affected by personal traits such as gender and ethnicity. While Large Language Models (LLMs) are widely used to simulate human responses across diverse contexts, their ability to account for demographic differences in subjective tasks remains uncertain. In this study, leveraging the POPQUORN dataset, we evaluate nine popular LLMs on their ability to understand demographic differences in two subjective judgment tasks: politeness and offensiveness. We find that in zero-shot settings, most models' predictions for both tasks align more closely with labels from White participants than those from Asian or Black participants, while only a minor gender bias favoring women appears in the politeness task. Furthermore, sociodemographic prompting does not consistently improve and, in some cases, worsens LLMs' ability to perceive language from specific sub-populations. These findings highlight potential demographic biases in LLMs when performing subjective judgment tasks and underscore the limitations of sociodemographic prompting as a strategy to achieve pluralistic alignment. Code and data are available at: https://github.com/Jiaxin-Pei/LLM-as-Subjective-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12217v1">Optimal Brain Iterative Merging: Mitigating Interference in LLM Merging</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities, but their high computational costs pose challenges for customization. Model merging offers a cost-effective alternative, yet existing methods suffer from interference among parameters, leading to performance degradation. In this work, we propose Optimal Brain Iterative Merging (OBIM), a novel method designed to mitigate both intra-model and inter-model interference. OBIM consists of two key components: (1) A saliency measurement mechanism that evaluates parameter importance based on loss changes induced by individual weight alterations, reducing intra-model interference by preserving only high-saliency parameters. (2) A mutually exclusive iterative merging framework, which incrementally integrates models using a binary mask to avoid direct parameter averaging, thereby mitigating inter-model interference. We validate OBIM through experiments on both Supervised Fine-Tuned (SFT) models and post-pretrained checkpoints. The results show that OBIM significantly outperforms existing merging techniques. Overall, OBIM provides an effective and practical solution for enhancing LLM merging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12216v1">Tactic: Adaptive Sparse Attention with Clustering and Distribution Fitting for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-02-17
    </div>
    <details class="paper-abstract">
      Long-context models are essential for many applications but face inefficiencies in loading large KV caches during decoding. Prior methods enforce fixed token budgets for sparse attention, assuming a set number of tokens can approximate full attention. However, these methods overlook variations in the importance of attention across heads, layers, and contexts. To address these limitations, we propose Tactic, a sparsity-adaptive and calibration-free sparse attention mechanism that dynamically selects tokens based on their cumulative attention scores rather than a fixed token budget. By setting a target fraction of total attention scores, Tactic ensures that token selection naturally adapts to variations in attention sparsity. To efficiently approximate this selection, Tactic leverages clustering-based sorting and distribution fitting, allowing it to accurately estimate token importance with minimal computational overhead. We show that Tactic outperforms existing sparse attention algorithms, achieving superior accuracy and up to 7.29x decode attention speedup. This improvement translates to an overall 1.58x end-to-end inference speedup, making Tactic a practical and effective solution for long-context LLM inference in accuracy-sensitive applications.
    </details>
</div>
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
