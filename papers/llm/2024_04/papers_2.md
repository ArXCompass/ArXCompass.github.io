# llm - 2024_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10896v1">From a Lossless (~1.5:1) Compression Algorithm for Llama2 7B Weights to Variable Precision, Variable Range, Compressed Numeric Data Types for CNNs and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      This paper starts with a simple lossless ~1.5:1 compression algorithm for the weights of the Large Language Model (LLM) Llama2 7B [1] that can be implemented in ~200 LUTs in AMD FPGAs, processing over 800 million bfloat16 numbers per second. This framework is then extended to variable precision, variable range, compressed numerical data types that are a user defined super set of both floats and posits [2]. The paper then discusses a simple hardware implementation of such format based on ANS (Asymmetrical Numeral Systems) [3] that acts as a bridge between this flexible data format and a computational engine while, at the same time, achieving bandwidth reduction. An example of a token factory using weight compression and sharing is also given.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.14972v3">Scaling Down to Scale Up: A Cost-Benefit Analysis of Replacing OpenAI's LLM with Open Source SLMs in Production</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 Updated title, Revised content
    </div>
    <details class="paper-abstract">
      Many companies use large language models (LLMs) offered as a service, like OpenAI's GPT-4, to create AI-enabled product experiences. Along with the benefits of ease-of-use and shortened time-to-solution, this reliance on proprietary services has downsides in model control, performance reliability, uptime predictability, and cost. At the same time, a flurry of open-source small language models (SLMs) has been made available for commercial use. However, their readiness to replace existing capabilities remains unclear, and a systematic approach to holistically evaluate these SLMs is not readily available. This paper presents a systematic evaluation methodology and a characterization of modern open-source SLMs and their trade-offs when replacing proprietary LLMs for a real-world product feature. We have designed SLaM, an open-source automated analysis tool that enables the quantitative and qualitative testing of product features utilizing arbitrary SLMs. Using SLaM, we examine the quality and performance characteristics of modern SLMs relative to an existing customer-facing implementation using the OpenAI GPT-4 API. Across 9 SLMs and their 29 variants, we observe that SLMs provide competitive results, significant performance consistency improvements, and a cost reduction of 5x~29x when compared to GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10757v1">Deep Learning and LLM-based Methods Applied to Stellar Lightcurve Classification</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 35 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Light curves serve as a valuable source of information on stellar formation and evolution. With the rapid advancement of machine learning techniques, it can be effectively processed to extract astronomical patterns and information. In this study, we present a comprehensive evaluation of deep-learning and large language model (LLM) based models for the automatic classification of variable star light curves, based on large datasets from the Kepler and K2 missions. Special emphasis is placed on Cepheids, RR Lyrae, and eclipsing binaries, examining the influence of observational cadence and phase distribution on classification precision. Employing AutoDL optimization, we achieve striking performance with the 1D-Convolution+BiLSTM architecture and the Swin Transformer, hitting accuracies of 94\% and 99\% correspondingly, with the latter demonstrating a notable 83\% accuracy in discerning the elusive Type II Cepheids-comprising merely 0.02\% of the total dataset.We unveil StarWhisper LightCurve (LC), an innovative Series comprising three LLM-based models: LLM, multimodal large language model (MLLM), and Large Audio Language Model (LALM). Each model is fine-tuned with strategic prompt engineering and customized training methods to explore the emergent abilities of these models for astronomical data. Remarkably, StarWhisper LC Series exhibit high accuracies around 90\%, significantly reducing the need for explicit feature engineering, thereby paving the way for streamlined parallel data processing and the progression of multifaceted multimodal models in astronomical applications. The study furnishes two detailed catalogs illustrating the impacts of phase and sampling intervals on deep learning classification accuracy, showing that a substantial decrease of up to 14\% in observation duration and 21\% in sampling points can be realized without compromising accuracy by more than 10\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10678v1">Automating REST API Postman Test Cases Using LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      In the contemporary landscape of technological advancements, the automation of manual processes is crucial, compelling the demand for huge datasets to effectively train and test machines. This research paper is dedicated to the exploration and implementation of an automated approach to generate test cases specifically using Large Language Models. The methodology integrates the use of Open AI to enhance the efficiency and effectiveness of test case generation for training and evaluating Large Language Models. This formalized approach with LLMs simplifies the testing process, making it more efficient and comprehensive. Leveraging natural language understanding, LLMs can intelligently formulate test cases that cover a broad range of REST API properties, ensuring comprehensive testing. The model that is developed during the research is trained using manually collected postman test cases or instances for various Rest APIs. LLMs enhance the creation of Postman test cases by automating the generation of varied and intricate test scenarios. Postman test cases offer streamlined automation, collaboration, and dynamic data handling, providing a user-friendly and efficient approach to API testing compared to traditional test cases. Thus, the model developed not only conforms to current technological standards but also holds the promise of evolving into an idea of substantial importance in future technological advancements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10261v4">How Good Are LLMs at Out-of-Distribution Detection?</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 Accepted at COLING 2024
    </div>
    <details class="paper-abstract">
      Out-of-distribution (OOD) detection plays a vital role in enhancing the reliability of machine learning (ML) models. The emergence of large language models (LLMs) has catalyzed a paradigm shift within the ML community, showcasing their exceptional capabilities across diverse natural language processing tasks. While existing research has probed OOD detection with relative small-scale Transformers like BERT, RoBERTa and GPT-2, the stark differences in scales, pre-training objectives, and inference paradigms call into question the applicability of these findings to LLMs. This paper embarks on a pioneering empirical investigation of OOD detection in the domain of LLMs, focusing on LLaMA series ranging from 7B to 65B in size. We thoroughly evaluate commonly-used OOD detectors, scrutinizing their performance in both zero-grad and fine-tuning scenarios. Notably, we alter previous discriminative in-distribution fine-tuning into generative fine-tuning, aligning the pre-training objective of LLMs with downstream tasks. Our findings unveil that a simple cosine distance OOD detector demonstrates superior efficacy, outperforming other OOD detectors. We provide an intriguing explanation for this phenomenon by highlighting the isotropic nature of the embedding spaces of LLMs, which distinctly contrasts with the anisotropic property observed in smaller BERT family models. The new insight enhances our understanding of how LLMs detect OOD data, thereby enhancing their adaptability and reliability in dynamic environments. We have released the source code at \url{https://github.com/Awenbocc/LLM-OOD} for other researchers to reproduce our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09151v2">Emerging Platforms Meet Emerging LLMs: A Year-Long Journey of Top-Down Development</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      Deploying machine learning (ML) on diverse computing platforms is crucial to accelerate and broaden their applications. However, it presents significant software engineering challenges due to the fast evolution of models, especially the recent Large Language Models (LLMs), and the emergence of new computing platforms. Current ML frameworks are primarily engineered for CPU and CUDA platforms, leaving a big gap in enabling emerging ones like Metal, Vulkan, and WebGPU. While a traditional bottom-up development pipeline fails to close the gap timely, we introduce TapML, a top-down approach and tooling designed to streamline the deployment of ML systems on diverse platforms, optimized for developer productivity. Unlike traditional bottom-up methods, which involve extensive manual testing and debugging, TapML automates unit testing through test carving and adopts a migration-based strategy for gradually offloading model computations from mature source platforms to emerging target platforms. By leveraging realistic inputs and remote connections for gradual target offloading, TapML accelerates the validation and minimizes debugging scopes, significantly optimizing development efforts. TapML was developed and applied through a year-long, real-world effort that successfully deployed significant emerging models and platforms. Through serious deployments of 82 emerging models in 17 distinct architectures across 5 emerging platforms, we showcase the effectiveness of TapML in enhancing developer productivity while ensuring model reliability and efficiency. Furthermore, we summarize comprehensive case studies from our real-world development, offering best practices for developing emerging ML systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07938v2">Large Language User Interfaces: Voice Interactive User Interfaces powered by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 Accepted as peer-reviewed publication
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Models (LLMs) has showcased remarkable capacities for logical reasoning and natural language comprehension. These capabilities can be leveraged in solutions that semantically and textually model complex problems. In this paper, we present our efforts toward constructing a framework that can serve as an intermediary between a user and their user interface (UI), enabling dynamic and real-time interactions. We employ a system that stands upon textual semantic mappings of UI components, in the form of annotations. These mappings are stored, parsed, and scaled in a custom data structure, supplementary to an agent-based prompting backend engine. Employing textual semantic mappings allows each component to not only explain its role to the engine but also provide expectations. By comprehending the needs of both the user and the components, our LLM engine can classify the most appropriate application, extract relevant parameters, and subsequently execute precise predictions of the user's expected actions. Such an integration evolves static user interfaces into highly dynamic and adaptable solutions, introducing a new frontier of intelligent and responsive user experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10308v1">Hierarchical Context Merging: Better Long Context Understanding for Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
      | 💬 Accepted to ICLR 2024. The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance in various natural language processing tasks. However, a primary constraint they face is the context limit, i.e., the maximum number of tokens they can process. Previous works have explored architectural changes and modifications in positional encoding to relax the constraint, but they often require expensive training or do not address the computational demands of self-attention. In this paper, we present Hierarchical cOntext MERging (HOMER), a new training-free scheme designed to overcome the limitations. HOMER uses a divide-and-conquer algorithm, dividing long inputs into manageable chunks. Each chunk is then processed collectively, employing a hierarchical strategy that merges adjacent chunks at progressive transformer layers. A token reduction technique precedes each merging, ensuring memory usage efficiency. We also propose an optimized computational order reducing the memory requirement to logarithmically scale with respect to input length, making it especially favorable for environments with tight memory restrictions. Our experiments demonstrate the proposed method's superior performance and memory efficiency, enabling the broader use of LLMs in contexts requiring extended context. Code is available at https://github.com/alinlab/HOMER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10304v1">LLM-Powered Test Case Generation for Detecting Tricky Bugs</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      Conventional automated test generation tools struggle to generate test oracles and tricky bug-revealing test inputs. Large Language Models (LLMs) can be prompted to produce test inputs and oracles for a program directly, but the precision of the tests can be very low for complex scenarios (only 6.3% based on our experiments). To fill this gap, this paper proposes AID, which combines LLMs with differential testing to generate fault-revealing test inputs and oracles targeting plausibly correct programs (i.e., programs that have passed all the existing tests). In particular, AID selects test inputs that yield diverse outputs on a set of program variants generated by LLMs, then constructs the test oracle based on the outputs. We evaluate AID on two large-scale datasets with tricky bugs: TrickyBugs and EvalPlus, and compare it with three state-of-the-art baselines. The evaluation results show that the recall, precision, and F1 score of AID outperform the state-of-the-art by up to 1.80x, 2.65x, and 1.66x, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.19102v3">Atom: Low-bit Quantization for Efficient and Accurate LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      The growing demand for Large Language Models (LLMs) in applications such as content generation, intelligent chatbots, and sentiment analysis poses considerable challenges for LLM service providers. To efficiently use GPU resources and boost throughput, batching multiple requests has emerged as a popular paradigm; to further speed up batching, LLM quantization techniques reduce memory consumption and increase computing capacity. However, prevalent quantization schemes (e.g., 8-bit weight-activation quantization) cannot fully leverage the capabilities of modern GPUs, such as 4-bit integer operators, resulting in sub-optimal performance. To maximize LLMs' serving throughput, we introduce Atom, a low-bit quantization method that achieves high throughput improvements with negligible accuracy loss. Atom significantly boosts serving throughput by using low-bit operators and considerably reduces memory consumption via low-bit quantization. It attains high accuracy by applying a novel mixed-precision and fine-grained quantization process. We evaluate Atom on 4-bit weight-activation quantization in the serving context. Atom improves end-to-end throughput (token/s) by up to $7.7\times$ compared to the FP16 and by $2.5\times$ compared to INT8 quantization, while maintaining the same latency target.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.10467v2">Stance Detection with Collaborative Role-Infused LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-16
    </div>
    <details class="paper-abstract">
      Stance detection automatically detects the stance in a text towards a target, vital for content analysis in web and social media research. Despite their promising capabilities, LLMs encounter challenges when directly applied to stance detection. First, stance detection demands multi-aspect knowledge, from deciphering event-related terminologies to understanding the expression styles in social media platforms. Second, stance detection requires advanced reasoning to infer authors' implicit viewpoints, as stance are often subtly embedded rather than overtly stated in the text. To address these challenges, we design a three-stage framework COLA (short for Collaborative rOle-infused LLM-based Agents) in which LLMs are designated distinct roles, creating a collaborative system where each role contributes uniquely. Initially, in the multidimensional text analysis stage, we configure the LLMs to act as a linguistic expert, a domain specialist, and a social media veteran to get a multifaceted analysis of texts, thus overcoming the first challenge. Next, in the reasoning-enhanced debating stage, for each potential stance, we designate a specific LLM-based agent to advocate for it, guiding the LLM to detect logical connections between text features and stance, tackling the second challenge. Finally, in the stance conclusion stage, a final decision maker agent consolidates prior insights to determine the stance. Our approach avoids extra annotated data and model training and is highly usable. We achieve state-of-the-art performance across multiple datasets. Ablation studies validate the effectiveness of each design role in handling stance detection. Further experiments have demonstrated the explainability and the versatility of our approach. Our approach excels in usability, accuracy, effectiveness, explainability and versatility, highlighting its value.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05800v2">Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 Accepted at NAACL 2024. Data released at https://github.com/google-research-datasets/swim-ir
    </div>
    <details class="paper-abstract">
      There has been limited success for dense retrieval models in multilingual retrieval, due to uneven and scarce training data available across multiple languages. Synthetic training data generation is promising (e.g., InPars or Promptagator), but has been investigated only for English. Therefore, to study model capabilities across both cross-lingual and monolingual retrieval tasks, we develop SWIM-IR, a synthetic retrieval training dataset containing 33 (high to very-low resource) languages for fine-tuning multilingual dense retrievers without requiring any human supervision. To construct SWIM-IR, we propose SAP (summarize-then-ask prompting), where the large language model (LLM) generates a textual summary prior to the query generation step. SAP assists the LLM in generating informative queries in the target language. Using SWIM-IR, we explore synthetic fine-tuning of multilingual dense retrieval models and evaluate them robustly on three retrieval benchmarks: XOR-Retrieve (cross-lingual), MIRACL (monolingual) and XTREME-UP (cross-lingual). Our models, called SWIM-X, are competitive with human-supervised dense retrieval models, e.g., mContriever-X, finding that SWIM-IR can cheaply substitute for expensive human-labeled retrieval training data. SWIM-IR dataset and SWIM-X models are available at https://github.com/google-research-datasets/SWIM-IR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10150v1">TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 Accepted to NAACL 2024 (long, main)
    </div>
    <details class="paper-abstract">
      Table reasoning is a challenging task that requires understanding both natural language questions and structured tabular data. Large language models (LLMs) have shown impressive capabilities in natural language understanding and generation, but they often struggle with large tables due to their limited input length. In this paper, we propose TabSQLify, a novel method that leverages text-to-SQL generation to decompose tables into smaller and relevant sub-tables, containing only essential information for answering questions or verifying statements, before performing the reasoning task. In our comprehensive evaluation on four challenging datasets, our approach demonstrates comparable or superior performance compared to prevailing methods reliant on full tables as input. Moreover, our method can reduce the input context length significantly, making it more scalable and efficient for large-scale table reasoning applications. Our method performs remarkably well on the WikiTQ benchmark, achieving an accuracy of 64.7%. Additionally, on the TabFact benchmark, it achieves a high accuracy of 79.5%. These results surpass other LLM-based baseline models on gpt-3.5-turbo (chatgpt). TabSQLify can reduce the table size significantly alleviating the computational load on LLMs when handling large tables without compromising performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10141v1">ANCHOR: LLM-driven News Subject Conditioning for Text-to-Image Synthesis</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 23 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Text-to-Image (T2I) Synthesis has made tremendous strides in enhancing synthesized image quality, but current datasets evaluate model performance only on descriptive, instruction-based prompts. Real-world news image captions take a more pragmatic approach, providing high-level situational and Named-Entity (NE) information and limited physical object descriptions, making them abstractive. To evaluate the ability of T2I models to capture intended subjects from news captions, we introduce the Abstractive News Captions with High-level cOntext Representation (ANCHOR) dataset, containing 70K+ samples sourced from 5 different news media organizations. With Large Language Models (LLM) achieving success in language and commonsense reasoning tasks, we explore the ability of different LLMs to identify and understand key subjects from abstractive captions. Our proposed method Subject-Aware Finetuning (SAFE), selects and enhances the representation of key subjects in synthesized images by leveraging LLM-generated subject weights. It also adapts to the domain distribution of news images and captions through custom Domain Fine-tuning, outperforming current T2I baselines on ANCHOR. By launching the ANCHOR dataset, we hope to motivate research in furthering the Natural Language Understanding (NLU) capabilities of T2I models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15834v1">FEDSTR: Money-In AI-Out | A Decentralized Marketplace for Federated Learning and LLM Training on the NOSTR Protocol</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      The NOSTR is a communication protocol for the social web, based on the w3c websockets standard. Although it is still in its infancy, it is well known as a social media protocol, thousands of trusted users and multiple user interfaces, offering a unique experience and enormous capabilities. To name a few, the NOSTR applications include but are not limited to direct messaging, file sharing, audio/video streaming, collaborative writing, blogging and data processing through distributed AI directories. In this work, we propose an approach that builds upon the existing protocol structure with end goal a decentralized marketplace for federated learning and LLM training. In this proposed design there are two parties: on one side there are customers who provide a dataset that they want to use for training an AI model. On the other side, there are service providers, who receive (parts of) the dataset, train the AI model, and for a payment as an exchange, they return the optimized AI model. The decentralized and censorship resistant features of the NOSTR enable the possibility of designing a fair and open marketplace for training AI models and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13076v1">LLM Evaluators Recognize and Favor Their Own Generations</a></div>
    <div class="paper-meta">
      📅 2024-04-15
    </div>
    <details class="paper-abstract">
      Self-evaluation using large language models (LLMs) has proven valuable not only in benchmarking but also methods like reward modeling, constitutional AI, and self-refinement. But new biases are introduced due to the same LLM acting as both the evaluator and the evaluatee. One such bias is self-preference, where an LLM evaluator scores its own outputs higher than others' while human annotators consider them of equal quality. But do LLMs actually recognize their own outputs when they give those texts higher scores, or is it just a coincidence? In this paper, we investigate if self-recognition capability contributes to self-preference. We discover that, out of the box, LLMs such as GPT-4 and Llama 2 have non-trivial accuracy at distinguishing themselves from other LLMs and humans. By fine-tuning LLMs, we discover a linear correlation between self-recognition capability and the strength of self-preference bias; using controlled experiments, we show that the causal explanation resists straightforward confounders. We discuss how self-recognition can interfere with unbiased evaluations and AI safety more generally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08669v2">On the Calibration of Multilingual Question Answering LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 Preprint. Under Submission
    </div>
    <details class="paper-abstract">
      Multilingual pre-trained Large Language Models (LLMs) are incredibly effective at Question Answering (QA), a core task in Natural Language Understanding, achieving high accuracies on several multilingual benchmarks. However, little is known about how well their confidences are calibrated. In this paper, we comprehensively benchmark the calibration of several multilingual LLMs (MLLMs) on a variety of QA tasks. We perform extensive experiments, spanning encoder-only, encoder-decoder, and decoder-only QA models (size varying from 110M to 7B parameters) and diverse languages, including both high- and low-resource ones. We study different dimensions of calibration in in-distribution, out-of-distribution, and cross-lingual transfer settings, and investigate strategies to improve it, including post-hoc methods and regularized fine-tuning. For decoder-only LLMs such as LlaMa2, we additionally find that in-context learning improves confidence calibration on multilingual data. We also conduct several ablation experiments to study the effect of language distances, language corpus size, and model size on calibration, and how multilingual models compare with their monolingual counterparts for diverse tasks and languages. Our experiments suggest that the multilingual QA models are poorly calibrated for languages other than English and incorporating a small set of cheaply translated multilingual samples during fine-tuning/calibration effectively enhances the calibration performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.08043v4">DiagGPT: An LLM-based and Multi-agent Dialogue System with Automatic Topic Management for Flexible Task-Oriented Dialogue</a></div>
    <div class="paper-meta">
      📅 2024-04-15
    </div>
    <details class="paper-abstract">
      A significant application of Large Language Models (LLMs), like ChatGPT, is their deployment as chat agents, which respond to human inquiries across a variety of domains. While current LLMs proficiently answer general questions, they often fall short in complex diagnostic scenarios such as legal, medical, or other specialized consultations. These scenarios typically require Task-Oriented Dialogue (TOD), where an AI chat agent must proactively pose questions and guide users toward specific goals or task completion. Previous fine-tuning models have underperformed in TOD and the full potential of conversational capability in current LLMs has not yet been fully explored. In this paper, we introduce DiagGPT (Dialogue in Diagnosis GPT), an innovative approach that extends LLMs to more TOD scenarios. In addition to guiding users to complete tasks, DiagGPT can effectively manage the status of all topics throughout the dialogue development. This feature enhances user experience and offers a more flexible interaction in TOD. Our experiments demonstrate that DiagGPT exhibits outstanding performance in conducting TOD with users, showing its potential for practical applications in various fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09492v1">Bridging the Gap between Different Vocabularies for LLM Ensemble</a></div>
    <div class="paper-meta">
      📅 2024-04-15
      | 💬 Accepted to the main conference of NAACL 2024
    </div>
    <details class="paper-abstract">
      Ensembling different large language models (LLMs) to unleash their complementary potential and harness their individual strengths is highly valuable. Nevertheless, vocabulary discrepancies among various LLMs have constrained previous studies to either selecting or blending completely generated outputs. This limitation hinders the dynamic correction and enhancement of outputs during the generation process, resulting in a limited capacity for effective ensemble. To address this issue, we propose a novel method to Ensemble LLMs via Vocabulary Alignment (EVA). EVA bridges the lexical gap among various LLMs, enabling meticulous ensemble at each generation step. Specifically, we first learn mappings between the vocabularies of different LLMs with the assistance of overlapping tokens. Subsequently, these mappings are employed to project output distributions of LLMs into a unified space, facilitating a fine-grained ensemble. Finally, we design a filtering strategy to exclude models that generate unfaithful tokens. Experimental results on commonsense reasoning, arithmetic reasoning, machine translation, and data-to-text generation tasks demonstrate the superiority of our approach compared with individual LLMs and previous ensemble methods conducted on complete outputs. Further analyses confirm that our approach can leverage knowledge from different language models and yield consistent improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.11872v2">Generation-driven Contrastive Self-training for Zero-shot Text Classification with Instruction-following LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-15
    </div>
    <details class="paper-abstract">
      The remarkable performance of large language models (LLMs) in zero-shot language understanding has garnered significant attention. However, employing LLMs for large-scale inference or domain-specific fine-tuning requires immense computational resources due to their substantial model size. To overcome these limitations, we introduce a novel method, namely GenCo, which leverages the strong generative power of LLMs to assist in training a smaller and more adaptable language model. In our method, an LLM plays an important role in the self-training loop of a smaller model in two important ways. Firstly, the LLM is used to augment each input instance with a variety of possible continuations, enriching its semantic context for better understanding. Secondly, it helps crafting additional high-quality training pairs, by rewriting input texts conditioned on predicted labels. This ensures the generated texts are highly relevant to the predicted labels, alleviating the prediction error during pseudo-labeling, while reducing the dependency on large volumes of unlabeled text. In our experiments, GenCo outperforms previous state-of-the-art methods when only limited ($<5\%$ of original) in-domain text data is available. Notably, our approach surpasses the performance of Alpaca-7B with human prompts, highlighting the potential of leveraging LLM for self-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.08126v1">Auctions with LLM Summaries</a></div>
    <div class="paper-meta">
      📅 2024-04-15
    </div>
    <details class="paper-abstract">
      We study an auction setting in which bidders bid for placement of their content within a summary generated by a large language model (LLM), e.g., an ad auction in which the display is a summary paragraph of multiple ads. This generalizes the classic ad settings such as position auctions to an LLM generated setting, which allows us to handle general display formats. We propose a novel factorized framework in which an auction module and an LLM module work together via a prediction model to provide welfare maximizing summary outputs in an incentive compatible manner. We provide a theoretical analysis of this framework and synthetic experiments to demonstrate the feasibility and validity of the system together with welfare comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09339v1">Towards Practical Tool Usage for Continually Learning LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-14
      | 💬 20 pages, 11 tables, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show an innate skill for solving language based tasks. But insights have suggested an inability to adjust for information or task-solving skills becoming outdated, as their knowledge, stored directly within their parameters, remains static in time. Tool use helps by offloading work to systems that the LLM can access through an interface, but LLMs that use them still must adapt to nonstationary environments for prolonged use, as new tools can emerge and existing tools can change. Nevertheless, tools require less specialized knowledge, therefore we hypothesize they are better suited for continual learning (CL) as they rely less on parametric memory for solving tasks and instead focus on learning when to apply pre-defined tools. To verify this, we develop a synthetic benchmark and follow this by aggregating existing NLP tasks to form a more realistic testing scenario. While we demonstrate scaling model size is not a solution, regardless of tool usage, continual learning techniques can enable tool LLMs to both adapt faster while forgetting less, highlighting their potential as continual learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.09135v1">Unveiling LLM Evaluation Focused on Metrics: Challenges and Solutions</a></div>
    <div class="paper-meta">
      📅 2024-04-14
    </div>
    <details class="paper-abstract">
      Natural Language Processing (NLP) is witnessing a remarkable breakthrough driven by the success of Large Language Models (LLMs). LLMs have gained significant attention across academia and industry for their versatile applications in text generation, question answering, and text summarization. As the landscape of NLP evolves with an increasing number of domain-specific LLMs employing diverse techniques and trained on various corpus, evaluating performance of these models becomes paramount. To quantify the performance, it's crucial to have a comprehensive grasp of existing metrics. Among the evaluation, metrics which quantifying the performance of LLMs play a pivotal role. This paper offers a comprehensive exploration of LLM evaluation from a metrics perspective, providing insights into the selection and interpretation of metrics currently in use. Our main goal is to elucidate their mathematical formulations and statistical interpretations. We shed light on the application of these metrics using recent Biomedical LLMs. Additionally, we offer a succinct comparison of these metrics, aiding researchers in selecting appropriate metrics for diverse tasks. The overarching goal is to furnish researchers with a pragmatic guide for effective LLM evaluation and metric selection, thereby advancing the understanding and application of these large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01444v3">Adapting LLM Agents with Universal Feedback in Communication</a></div>
    <div class="paper-meta">
      📅 2024-04-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated potential for LLM agents. To facilitate the training for these agents with both linguistic feedback and non-linguistic reward signals, we introduce Learning through Communication (LTC). We design a universal buffer to store all the feedback, and an iterative pipeline to enable an LLM agent to explore and update its policy in an given environment. To optimize agent interactions for task-specific learning with our universal buffer and pipeline, we introduce diverse communication patterns tailored for both single-agent and multi-agent environments. We evaluate the efficacy of our LTC approach on four diverse datasets: ALFWorld (single-agent), HotpotQA (multi-agent collaboration), Chameleon (multi-agent competition), and GSM8k (multi-agent teacher-student). On these data sets, LTC outperforms the supervised instruction fine-tuning baselines by 3.6% to 12%. These results highlight the versatility and efficiency of LTC in facilitating online adaptation for LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00785v4">BooookScore: A systematic exploration of book-length summarization in the era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-13
      | 💬 ICLR 2024 camera-ready (updated figure1 and table2; corrected minor details in the explanation of hierarchical merging)
    </div>
    <details class="paper-abstract">
      Summarizing book-length documents (>100K tokens) that exceed the context window size of large language models (LLMs) requires first breaking the input document into smaller chunks and then prompting an LLM to merge, update, and compress chunk-level summaries. Despite the complexity and importance of this task, it has yet to be meaningfully studied due to the challenges of evaluation: existing book-length summarization datasets (e.g., BookSum) are in the pretraining data of most public LLMs, and existing evaluation methods struggle to capture errors made by modern LLM summarizers. In this paper, we present the first study of the coherence of LLM-based book-length summarizers implemented via two prompting workflows: (1) hierarchically merging chunk-level summaries, and (2) incrementally updating a running summary. We obtain 1193 fine-grained human annotations on GPT-4 generated summaries of 100 recently-published books and identify eight common types of coherence errors made by LLMs. Because human evaluation is expensive and time-consuming, we develop an automatic metric, BooookScore, that measures the proportion of sentences in a summary that do not contain any of the identified error types. BooookScore has high agreement with human annotations and allows us to systematically evaluate the impact of many other critical parameters (e.g., chunk size, base LLM) while saving $15K USD and 500 hours in human evaluation costs. We find that closed-source LLMs such as GPT-4 and Claude 2 produce summaries with higher BooookScore than those generated by open-source models. While LLaMA 2 falls behind other models, Mixtral achieves performance on par with GPT-3.5-Turbo. Incremental updating yields lower BooookScore but higher level of detail than hierarchical merging, a trade-off sometimes preferred by annotators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17963v2">M$^{2}$Chat: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-13
    </div>
    <details class="paper-abstract">
      While current LLM chatbots like GPT-4V bridge the gap between human instructions and visual representations to enable text-image generations, they still lack efficient alignment methods for high-fidelity performance on multiple downstream tasks. In this paper, we propose \textbf{$M^{2}Chat$}, a novel unified multimodal LLM framework for generating interleaved text-image conversation across various scenarios. Specifically, we propose an $M^{3}Adapter$ that efficiently integrates granular low-level visual information and high-level semantic features from multi-modality prompts. Upon the well-aligned fused feature, $M^{3}Adapter$ tailors a learnable gating strategy to balance the model creativity and consistency across various tasks adaptively. Moreover, to further enhance the effectiveness of $M^{3}Adapter$ while preserving the coherence of semantic context comprehension, we introduce a two-stage $M^{3}FT$ fine-tuning strategy. This strategy optimizes disjoint groups of parameters for image-text alignment and visual-instruction respectively. Extensive experiments demonstrate our $M^{2}Chat$ surpasses state-of-the-art counterparts across diverse benchmarks, showcasing its prowess in interleaving generation, storytelling, and multimodal dialogue systems. The demo and code are available at \red{https://mattie-e.github.io/M2Chat.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08886v1">EIVEN: Efficient Implicit Attribute Value Extraction using Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-04-13
      | 💬 Accepted by NAACL 2024 Industry Track
    </div>
    <details class="paper-abstract">
      In e-commerce, accurately extracting product attribute values from multimodal data is crucial for improving user experience and operational efficiency of retailers. However, previous approaches to multimodal attribute value extraction often struggle with implicit attribute values embedded in images or text, rely heavily on extensive labeled data, and can easily confuse similar attribute values. To address these issues, we introduce EIVEN, a data- and parameter-efficient generative framework that pioneers the use of multimodal LLM for implicit attribute value extraction. EIVEN leverages the rich inherent knowledge of a pre-trained LLM and vision encoder to reduce reliance on labeled data. We also introduce a novel Learning-by-Comparison technique to reduce model confusion by enforcing attribute value comparison and difference identification. Additionally, we construct initial open-source datasets for multimodal implicit attribute value extraction. Our extensive experiments reveal that EIVEN significantly outperforms existing methods in extracting implicit attribute values while requiring less labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08865v1">LLM In-Context Recall is Prompt Dependent</a></div>
    <div class="paper-meta">
      📅 2024-04-13
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) highlights the critical importance of conducting thorough evaluations to discern their comparative advantages, limitations, and optimal use cases. Particularly important is assessing their capacity to accurately retrieve information included in a given prompt. A model's ability to do this significantly influences how effectively it can utilize contextual details, thus impacting its practical efficacy and dependability in real-world applications. Our research analyzes the in-context recall performance of various LLMs using the needle-in-a-haystack method. In this approach, a factoid (the "needle") is embedded within a block of filler text (the "haystack"), which the model is asked to retrieve. We assess the recall performance of each model across various haystack lengths and with varying needle placements to identify performance patterns. This study demonstrates that an LLM's recall capability is not only contingent upon the prompt's content but also may be compromised by biases in its training data. Conversely, adjustments to model architecture, training strategy, or fine-tuning can improve performance. Our analysis provides insight into LLM behavior, offering direction for the development of more effective applications of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01643v2">L-TUNING: Synchronized Label Tuning for Prompt and Prefix in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-13
      | 💬 Published in the ICLR TinyPaper track
    </div>
    <details class="paper-abstract">
      Efficiently fine-tuning Large Language Models (LLMs) for specific tasks presents a considerable challenge in natural language processing. Traditional methods, like prompt or prefix tuning, typically rely on arbitrary tokens for training, leading to prolonged training times and generalized token use across various class labels. To address these issues, this paper introduces L-Tuning, an efficient fine-tuning approach designed for classification tasks within the Natural Language Inference (NLI) framework. Diverging from conventional methods, L-Tuning focuses on the fine-tuning of label tokens processed through a pre-trained LLM, thereby harnessing its pre-existing semantic knowledge. This technique not only improves the fine-tuning accuracy and efficiency but also facilitates the generation of distinct label embeddings for each class, enhancing the model's training nuance. Our experimental results indicate a significant improvement in training efficiency and classification accuracy with L-Tuning compared to traditional approaches, marking a promising advancement in fine-tuning LLMs for complex language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08806v1">CreativEval: Evaluating Creativity of LLM-Based Hardware Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have proved effective and efficient in generating code, leading to their utilization within the hardware design process. Prior works evaluating LLMs' abilities for register transfer level code generation solely focus on functional correctness. However, the creativity associated with these LLMs, or the ability to generate novel and unique solutions, is a metric not as well understood, in part due to the challenge of quantifying this quality. To address this research gap, we present CreativeEval, a framework for evaluating the creativity of LLMs within the context of generating hardware designs. We quantify four creative sub-components, fluency, flexibility, originality, and elaboration, through various prompting and post-processing techniques. We then evaluate multiple popular LLMs (including GPT models, CodeLlama, and VeriGen) upon this creativity metric, with results indicating GPT-3.5 as the most creative model in generating hardware designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06753v2">AudioChatLlama: Towards General-Purpose Speech Abilities for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-12
    </div>
    <details class="paper-abstract">
      In this work, we extend the instruction-tuned Llama-2 model with end-to-end general-purpose speech processing and reasoning abilities while maintaining the wide range of original LLM capabilities, without using any carefully curated paired data. The resulting end-to-end model, named AudioChatLlama, can utilize audio prompts as a replacement for text and sustain a conversation. Such a model also has extended cross-modal capabilities such as being able to perform spoken question answering (QA), speech translation, and audio summarization amongst many other closed and open-domain tasks. This is unlike prior approaches in speech, in which LLMs are extended to handle audio for a limited number of pre-designated tasks. On both synthesized and recorded speech QA test sets, evaluations show that our end-to-end approach is on par with or outperforms cascaded systems (speech recognizer + LLM) in terms of modeling the response to a prompt. Furthermore, unlike cascades, our approach can interchange text and audio modalities and intrinsically utilize prior context in a conversation to provide better results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08727v1">Can LLMs substitute SQL? Comparing Resource Utilization of Querying LLMs versus Traditional Relational Databases</a></div>
    <div class="paper-meta">
      📅 2024-04-12
      | 💬 13 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can automate or substitute different types of tasks in the software engineering process. This study evaluates the resource utilization and accuracy of LLM in interpreting and executing natural language queries against traditional SQL within relational database management systems. We empirically examine the resource utilization and accuracy of nine LLMs varying from 7 to 34 Billion parameters, including Llama2 7B, Llama2 13B, Mistral, Mixtral, Optimus-7B, SUS-chat-34B, platypus-yi-34b, NeuralHermes-2.5-Mistral-7B and Starling-LM-7B-alpha, using a small transaction dataset. Our findings indicate that using LLMs for database queries incurs significant energy overhead (even small and quantized models), making it an environmentally unfriendly approach. Therefore, we advise against replacing relational databases with LLMs due to their substantial resource utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08517v1">Online Safety Analysis for LLMs: a Benchmark, an Assessment, and a Path Forward</a></div>
    <div class="paper-meta">
      📅 2024-04-12
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have seen widespread applications across numerous fields, their limited interpretability poses concerns regarding their safe operations from multiple aspects, e.g., truthfulness, robustness, and fairness. Recent research has started developing quality assurance methods for LLMs, introducing techniques such as offline detector-based or uncertainty estimation methods. However, these approaches predominantly concentrate on post-generation analysis, leaving the online safety analysis for LLMs during the generation phase an unexplored area. To bridge this gap, we conduct in this work a comprehensive evaluation of the effectiveness of existing online safety analysis methods on LLMs. We begin with a pilot study that validates the feasibility of detecting unsafe outputs in the early generation process. Following this, we establish the first publicly available benchmark of online safety analysis for LLMs, including a broad spectrum of methods, models, tasks, datasets, and evaluation metrics. Utilizing this benchmark, we extensively analyze the performance of state-of-the-art online safety analysis methods on both open-source and closed-source LLMs. This analysis reveals the strengths and weaknesses of individual methods and offers valuable insights into selecting the most appropriate method based on specific application scenarios and task requirements. Furthermore, we also explore the potential of using hybridization methods, i.e., combining multiple methods to derive a collective safety conclusion, to enhance the efficacy of online safety analysis for LLMs. Our findings indicate a promising direction for the development of innovative and trustworthy quality assurance methodologies for LLMs, facilitating their reliable deployments across diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04643v2">QAQ: Quality Adaptive Quantization for LLM KV Cache</a></div>
    <div class="paper-meta">
      📅 2024-04-12
    </div>
    <details class="paper-abstract">
      The emergence of LLMs has ignited a fresh surge of breakthroughs in NLP applications, particularly in domains such as question-answering systems and text generation. As the need for longer context grows, a significant bottleneck in model deployment emerges due to the linear expansion of the Key-Value (KV) cache with the context length. Existing methods primarily rely on various hypotheses, such as sorting the KV cache based on attention scores for replacement or eviction, to compress the KV cache and improve model throughput. However, heuristics used by these strategies may wrongly evict essential KV cache, which can significantly degrade model performance. In this paper, we propose QAQ, a Quality Adaptive Quantization scheme for the KV cache. We theoretically demonstrate that key cache and value cache exhibit distinct sensitivities to quantization, leading to the formulation of separate quantization strategies for their non-uniform quantization. Through the integration of dedicated outlier handling, as well as an improved attention-aware approach, QAQ achieves up to 10x the compression ratio of the KV cache size with a neglectable impact on model performance. QAQ significantly reduces the practical hurdles of deploying LLMs, opening up new possibilities for longer-context applications. The code is available at github.com/ClubieDong/KVCacheQuantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08417v1">AdapterSwap: Continuous Training of LLMs with Data Removal and Access-Control Guarantees</a></div>
    <div class="paper-meta">
      📅 2024-04-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly capable of completing knowledge intensive tasks by recalling information from a static pretraining corpus. Here we are concerned with LLMs in the context of evolving data requirements. For instance: batches of new data that are introduced periodically; subsets of data with user-based access controls; or requirements on dynamic removal of documents with guarantees that associated knowledge cannot be recalled. We wish to satisfy these requirements while at the same time ensuring a model does not forget old information when new data becomes available. To address these issues, we introduce AdapterSwap, a training and inference scheme that organizes knowledge from a data collection into a set of low-rank adapters, which are dynamically composed during inference. Our experiments demonstrate AdapterSwap's ability to support efficient continual learning, while also enabling organizations to have fine-grained control over data access and deletion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17097v2">Re-Ex: Revising after Explanation Reduces the Factual Errors in LLM Responses</a></div>
    <div class="paper-meta">
      📅 2024-04-12
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Mitigating hallucination issues is a key challenge that must be overcome to reliably deploy large language models (LLMs) in real-world scenarios. Recently, various methods have been proposed to detect and revise factual errors in LLM-generated texts, in order to reduce hallucination. In this paper, we propose Re-Ex, a method for post-editing LLM-generated responses. Re-Ex introduces a novel reasoning step dubbed as the factual error explanation step. Re-Ex revises the initial response of LLMs using 3-steps : first, external tools are used to retrieve the evidences of the factual errors in the initial LLM response; next, LLM is instructed to explain the problematic parts of the response based on the gathered evidence; finally, LLM revises the initial response using the explanations provided in the previous step. In addition to the explanation step, Re-Ex also incorporates new prompting techniques to reduce the token count and inference time required for the response revision process. Compared with existing methods including FacTool, CoVE, and RARR, Re-Ex provides better detection and revision performance with less inference time and fewer tokens in multiple benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08772v2">HuixiangDou: Overcoming Group Chat Scenarios with LLM-based Technical Assistance</a></div>
    <div class="paper-meta">
      📅 2024-04-12
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In this work, we present HuixiangDou, a technical assistant powered by Large Language Models (LLM). This system is designed to assist algorithm developers by providing insightful responses to questions related to open-source algorithm projects, such as computer vision and deep learning projects from OpenMMLab. We further explore the integration of this assistant into the group chats of instant messaging (IM) tools such as WeChat and Lark. Through several iterative improvements and trials, we have developed a sophisticated technical chat assistant capable of effectively answering users' technical questions without causing message flooding. This paper's contributions include: 1) Designing an algorithm pipeline specifically for group chat scenarios; 2) Verifying the reliable performance of text2vec in task rejection; 3) Identifying three critical requirements for LLMs in technical-assistant-like products, namely scoring ability, In-Context Learning (ICL), and Long Context. We have made the source code, android app and web service available at Github (https://github.com/internlm/huixiangdou), OpenXLab (https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web) and YouTube (https://youtu.be/ylXrT-Tei-Y) to aid in future research and application. HuixiangDou is applicable to any group chat within IM tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08309v1">Subtoxic Questions: Dive Into Attitude Change of LLM's Response in Jailbreak Attempts</a></div>
    <div class="paper-meta">
      📅 2024-04-12
      | 💬 4 pages, 2 figures. This paper was submitted to The 7th Deep Learning Security and Privacy Workshop (DLSP 2024) and was accepted as extended abstract, see https://dlsp2024.ieee-security.org/
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) of Prompt Jailbreaking are getting more and more attention, it is of great significance to raise a generalized research paradigm to evaluate attack strengths and a basic model to conduct subtler experiments. In this paper, we propose a novel approach by focusing on a set of target questions that are inherently more sensitive to jailbreak prompts, aiming to circumvent the limitations posed by enhanced LLM security. Through designing and analyzing these sensitive questions, this paper reveals a more effective method of identifying vulnerabilities in LLMs, thereby contributing to the advancement of LLM security. This research not only challenges existing jailbreaking methodologies but also fortifies LLMs against potential exploits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01246v2">LimSim++: A Closed-Loop Platform for Deploying Multimodal LLMs in Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2024-04-12
      | 💬 Accepted by 35th IEEE Intelligent Vehicles Symposium (IV 2024)
    </div>
    <details class="paper-abstract">
      The emergence of Multimodal Large Language Models ((M)LLMs) has ushered in new avenues in artificial intelligence, particularly for autonomous driving by offering enhanced understanding and reasoning capabilities. This paper introduces LimSim++, an extended version of LimSim designed for the application of (M)LLMs in autonomous driving. Acknowledging the limitations of existing simulation platforms, LimSim++ addresses the need for a long-term closed-loop infrastructure supporting continuous learning and improved generalization in autonomous driving. The platform offers extended-duration, multi-scenario simulations, providing crucial information for (M)LLM-driven vehicles. Users can engage in prompt engineering, model evaluation, and framework enhancement, making LimSim++ a versatile tool for research and practice. This paper additionally introduces a baseline (M)LLM-driven framework, systematically validated through quantitative experiments across diverse scenarios. The open-source resources of LimSim++ are available at: https://pjlab-adg.github.io/limsim-plus/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08148v1">Distilling Algorithmic Reasoning from LLMs via Explaining Solution Programs</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 pre-print
    </div>
    <details class="paper-abstract">
      Distilling explicit chain-of-thought reasoning paths has emerged as an effective method for improving the reasoning abilities of large language models (LLMs) across various tasks. However, when tackling complex tasks that pose significant challenges for state-of-the-art models, this technique often struggles to produce effective chains of thought that lead to correct answers. In this work, we propose a novel approach to distill reasoning abilities from LLMs by leveraging their capacity to explain solutions. We apply our method to solving competitive-level programming challenges. More specifically, we employ an LLM to generate explanations for a set of <problem, solution-program> pairs, then use <problem, explanation> pairs to fine-tune a smaller language model, which we refer to as the Reasoner, to learn algorithmic reasoning that can generate "how-to-solve" hints for unseen problems. Our experiments demonstrate that learning from explanations enables the Reasoner to more effectively guide program implementation by a Coder, resulting in higher solve rates than strong chain-of-thought baselines on competitive-level programming problems. It also outperforms models that learn directly from <problem, solution-program> pairs. We curated an additional test set in the CodeContests format, which includes 246 more recent problems posted after the models' knowledge cutoff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08126v1">Auctions with LLM Summaries</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      We study an auction setting in which bidders bid for placement of their content within a summary generated by a large language model (LLM), e.g., an ad auction in which the display is a summary paragraph of multiple ads. This generalizes the classic ad settings such as position auctions to an LLM generated setting, which allows us to handle general display formats. We propose a novel factorized framework in which an auction module and an LLM module work together via a prediction model to provide welfare maximizing summary outputs in an incentive compatible manner. We provide a theoretical analysis of this framework and synthetic experiments to demonstrate the feasibility and validity of the system together with welfare comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08092v1">Data-Augmentation-Based Dialectal Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      This report presents GMUNLP's participation to the Dialect-Copa shared task at VarDial 2024, which focuses on evaluating the commonsense reasoning capabilities of large language models (LLMs) on South Slavic micro-dialects. The task aims to assess how well LLMs can handle non-standard dialectal varieties, as their performance on standard languages is already well-established. We propose an approach that combines the strengths of different types of language models and leverages data augmentation techniques to improve task performance on three South Slavic dialects: Chakavian, Cherkano, and Torlak. We conduct experiments using a language-family-focused encoder-based model (BERTi\'c) and a domain-agnostic multilingual model (AYA-101). Our results demonstrate that the proposed data augmentation techniques lead to substantial performance gains across all three test datasets in the open-source model category. This work highlights the practical utility of data augmentation and the potential of LLMs in handling non-standard dialectal varieties, contributing to the broader goal of advancing natural language understanding in low-resource and dialectal settings. Code:https://github.com/ffaisal93/dialect_copa
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08078v1">SQBC: Active Learning using LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      Stance detection is an important task for many applications that analyse or support online political discussions. Common approaches include fine-tuning transformer based models. However, these models require a large amount of labelled data, which might not be available. In this work, we present two different ways to leverage LLM-generated synthetic data to train and improve stance detection agents for online political discussions: first, we show that augmenting a small fine-tuning dataset with synthetic data can improve the performance of the stance detection model. Second, we propose a new active learning method called SQBC based on the "Query-by-Comittee" approach. The key idea is to use LLM-generated synthetic data as an oracle to identify the most informative unlabelled samples, that are selected for manual labelling. Comprehensive experiments show that both ideas can improve the stance detection performance. Curiously, we observed that fine-tuning on actively selected samples can exceed the performance of using the full dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07926v1">Leveraging Large Language Models (LLMs) to Support Collaborative Human-AI Online Risk Data Annotation</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 This paper has been peer-reviewed and presented at the "CHI 2024 Workshop on LLMs as Research Tools: Applications and Evaluations in HCI Data Work, May 12, 2024, Honolulu, HI, USA."
    </div>
    <details class="paper-abstract">
      In this position paper, we discuss the potential for leveraging LLMs as interactive research tools to facilitate collaboration between human coders and AI to effectively annotate online risk data at scale. Collaborative human-AI labeling is a promising approach to annotating large-scale and complex data for various tasks. Yet, tools and methods to support effective human-AI collaboration for data annotation are under-studied. This gap is pertinent because co-labeling tasks need to support a two-way interactive discussion that can add nuance and context, particularly in the context of online risk, which is highly subjective and contextualized. Therefore, we provide some of the early benefits and challenges of using LLMs-based tools for risk annotation and suggest future directions for the HCI research community to leverage LLMs as research tools to facilitate human-AI collaboration in contextualized online data annotation. Our research interests align very well with the purposes of the LLMs as Research Tools workshop to identify ongoing applications and challenges of using LLMs to work with data in HCI research. We anticipate learning valuable insights from organizers and participants into how LLMs can help reshape the HCI community's methods for working with data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06948v2">MetaCheckGPT -- A Multi-task Hallucination Detector Using LLM Uncertainty and Meta-models</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 Entry for SemEval-2024 Shared Task 6: SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) have recently become a significant problem. A recent effort in this direction is a shared task at Semeval 2024 Task 6, SHROOM, a Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. This paper describes our winning solution ranked 1st and 2nd in the 2 sub-tasks of model agnostic and model aware tracks respectively. We propose a meta-regressor framework of LLMs for model evaluation and integration that achieves the highest scores on the leaderboard. We also experiment with various transformer-based models and black box methods like ChatGPT, Vectara, and others. In addition, we perform an error analysis comparing GPT4 against our best model which shows the limitations of the former.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14785v2">Simple Linguistic Inferences of Large Language Models (LLMs): Blind Spots and Blinds</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      We evaluate LLMs' language understanding capacities on simple inference tasks that most humans find trivial. Specifically, we target (i) grammatically-specified entailments, (ii) premises with evidential adverbs of uncertainty, and (iii) monotonicity entailments. We design evaluation sets for these tasks and conduct experiments in both zero-shot and chain-of-thought setups, and with multiple prompts and LLMs. The models exhibit moderate to low performance on these evaluation sets. Subsequent experiments show that embedding the premise in syntactic constructions that should preserve the entailment relations (presupposition triggers) or change them (non-factives), further confuses the models, causing them to either under-predict or over-predict certain entailment labels regardless of the true relation, and often disregarding the nature of the embedding context. Overall these results suggest that, despite LLMs' celebrated language understanding capacity, even the strongest models have blindspots with respect to certain types of entailments, and certain information-packaging structures act as ``blinds'' overshadowing the semantics of the embedded premise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.01957v3">Distilled Self-Critique of LLMs with Synthetic Data: a Bayesian Perspective</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 Accepted to ICLR 2024 (TinyPapers track)
    </div>
    <details class="paper-abstract">
      This paper proposes an interpretation of RLAIF as Bayesian inference by introducing distilled Self-Critique (dSC), which refines the outputs of a LLM through a Gibbs sampler that is later distilled into a fine-tuned model. Only requiring synthetic data, dSC is exercised in experiments regarding safety, sentiment, and privacy control, showing it can be a viable and cheap alternative to align LLMs. Code released at \url{https://github.com/vicgalle/distilled-self-critique}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07613v1">Medical mT5: An Open-Source Multilingual Text-to-Text LLM for The Medical Domain</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 LREC-COLING 2024
    </div>
    <details class="paper-abstract">
      Research on language technology for the development of medical applications is currently a hot topic in Natural Language Understanding and Generation. Thus, a number of large language models (LLMs) have recently been adapted to the medical domain, so that they can be used as a tool for mediating in human-AI interaction. While these LLMs display competitive performance on automated medical texts benchmarks, they have been pre-trained and evaluated with a focus on a single language (English mostly). This is particularly true of text-to-text models, which typically require large amounts of domain-specific pre-training data, often not easily accessible for many languages. In this paper, we address these shortcomings by compiling, to the best of our knowledge, the largest multilingual corpus for the medical domain in four languages, namely English, French, Italian and Spanish. This new corpus has been used to train Medical mT5, the first open-source text-to-text multilingual model for the medical domain. Additionally, we present two new evaluation benchmarks for all four languages with the aim of facilitating multilingual research in this domain. A comprehensive evaluation shows that Medical mT5 outperforms both encoders and similarly sized text-to-text models for the Spanish, French, and Italian benchmarks, while being competitive with current state-of-the-art LLMs in English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06845v2">DriveDreamer-2: LLM-Enhanced World Models for Diverse Driving Video Generation</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 Project Page: https://drivedreamer2.github.io
    </div>
    <details class="paper-abstract">
      World models have demonstrated superiority in autonomous driving, particularly in the generation of multi-view driving videos. However, significant challenges still exist in generating customized driving videos. In this paper, we propose DriveDreamer-2, which builds upon the framework of DriveDreamer and incorporates a Large Language Model (LLM) to generate user-defined driving videos. Specifically, an LLM interface is initially incorporated to convert a user's query into agent trajectories. Subsequently, a HDMap, adhering to traffic regulations, is generated based on the trajectories. Ultimately, we propose the Unified Multi-View Model to enhance temporal and spatial coherence in the generated driving videos. DriveDreamer-2 is the first world model to generate customized driving videos, it can generate uncommon driving videos (e.g., vehicles abruptly cut in) in a user-friendly manner. Besides, experimental results demonstrate that the generated videos enhance the training of driving perception methods (e.g., 3D detection and tracking). Furthermore, video generation quality of DriveDreamer-2 surpasses other state-of-the-art methods, showcasing FID and FVD scores of 11.2 and 55.7, representing relative improvements of 30% and 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07456v1">WESE: Weak Exploration to Strong Exploitation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated remarkable potential as an intelligent agent. However, existing researches mainly focus on enhancing the agent's reasoning or decision-making abilities through well-designed prompt engineering or task-specific fine-tuning, ignoring the procedure of exploration and exploitation. When addressing complex tasks within open-world interactive environments, these methods exhibit limitations. Firstly, the lack of global information of environments leads to greedy decisions, resulting in sub-optimal solutions. On the other hand, irrelevant information acquired from the environment not only adversely introduces noise, but also incurs additional cost. This paper proposes a novel approach, Weak Exploration to Strong Exploitation (WESE), to enhance LLM agents in solving open-world interactive tasks. Concretely, WESE involves decoupling the exploration and exploitation process, employing a cost-effective weak agent to perform exploration tasks for global knowledge. A knowledge graph-based strategy is then introduced to store the acquired knowledge and extract task-relevant knowledge, enhancing the stronger agent in success rate and efficiency for the exploitation task. Our approach is flexible enough to incorporate diverse tasks, and obtains significant improvements in both success rates and efficiency across four interactive benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07449v1">Learning to Localize Objects Improves Spatial Reasoning in Visual-LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      Integration of Large Language Models (LLMs) into visual domain tasks, resulting in visual-LLMs (V-LLMs), has enabled exceptional performance in vision-language tasks, particularly for visual question answering (VQA). However, existing V-LLMs (e.g. BLIP-2, LLaVA) demonstrate weak spatial reasoning and localization awareness. Despite generating highly descriptive and elaborate textual answers, these models fail at simple tasks like distinguishing a left vs right location. In this work, we explore how image-space coordinate based instruction fine-tuning objectives could inject spatial awareness into V-LLMs. We discover optimal coordinate representations, data-efficient instruction fine-tuning objectives, and pseudo-data generation strategies that lead to improved spatial awareness in V-LLMs. Additionally, our resulting model improves VQA across image and video domains, reduces undesired hallucination, and generates better contextual object descriptions. Experiments across 5 vision-language tasks involving 14 different datasets establish the clear performance improvements achieved by our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11134v2">OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 In SIGIR 2024 Resource & Reproducibility Track
    </div>
    <details class="paper-abstract">
      In recent years, the integration of Large Language Models (LLMs) into recommender systems has garnered interest among both practitioners and researchers. Despite this interest, the field is still emerging, and the lack of open-source R&D platforms may impede the exploration of LLM-based recommendations. This paper introduces OpenP5, an open-source platform designed as a resource to facilitate the development, training, and evaluation of LLM-based generative recommender systems for research purposes. The platform is implemented using encoder-decoder LLMs (e.g., T5) and decoder-only LLMs (e.g., Llama-2) across 10 widely recognized public datasets, catering to two fundamental recommendation tasks: sequential and straightforward recommendations. Recognizing the crucial role of item IDs in LLM-based recommendations, we have also incorporated three item indexing methods within the OpenP5 platform: random indexing, sequential indexing and collaborative indexing. Built on the Transformers library, the platform facilitates easy customization of LLM-based recommendations for users. OpenP5 boasts a range of features including extensible data processing, task-centric optimization, comprehensive datasets and checkpoints, efficient acceleration, and standardized evaluations, making it a valuable tool for the implementation and evaluation of LLM-based recommender systems. The open-source code and pre-trained checkpoints for the OpenP5 library are publicly available at https://github.com/agiresearch/OpenP5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07108v2">From Model-centered to Human-Centered: Revision Distance as a Metric for Text Evaluation in LLMs-based Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-11
      | 💬 9 pages, 2 figures, under review
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) is fundamental, particularly in the context of practical applications. Conventional evaluation methods, typically designed primarily for LLM development, yield numerical scores that ignore the user experience. Therefore, our study shifts the focus from model-centered to human-centered evaluation in the context of AI-powered writing assistance applications. Our proposed metric, termed ``Revision Distance,'' utilizes LLMs to suggest revision edits that mimic the human writing process. It is determined by counting the revision edits generated by LLMs. Benefiting from the generated revision edit details, our metric can provide a self-explained text evaluation result in a human-understandable manner beyond the context-independent score. Our results show that for the easy-writing task, ``Revision Distance'' is consistent with established metrics (ROUGE, Bert-score, and GPT-score), but offers more insightful, detailed feedback and better distinguishes between texts. Moreover, in the context of challenging academic writing tasks, our metric still delivers reliable evaluations where other metrics tend to struggle. Furthermore, our metric also holds significant potential for scenarios lacking reference texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.08126v1">Auctions with LLM Summaries</a></div>
    <div class="paper-meta">
      📅 2024-04-11
    </div>
    <details class="paper-abstract">
      We study an auction setting in which bidders bid for placement of their content within a summary generated by a large language model (LLM), e.g., an ad auction in which the display is a summary paragraph of multiple ads. This generalizes the classic ad settings such as position auctions to an LLM generated setting, which allows us to handle general display formats. We propose a novel factorized framework in which an auction module and an LLM module work together via a prediction model to provide welfare maximizing summary outputs in an incentive compatible manner. We provide a theoretical analysis of this framework and synthetic experiments to demonstrate the feasibility and validity of the system together with welfare comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05326v4">ChatASU: Evoking LLM's Reflexion to Truly Understand Aspect Sentiment in Dialogues</a></div>
    <div class="paper-meta">
      📅 2024-04-10
    </div>
    <details class="paper-abstract">
      Aspect Sentiment Understanding (ASU) in interactive scenarios (e.g., Question-Answering and Dialogue) has attracted ever-more interest in recent years and achieved important progresses. However, existing studies on interactive ASU largely ignore the coreference issue for opinion targets (i.e., aspects), while this phenomenon is ubiquitous in interactive scenarios especially dialogues, limiting the ASU performance. Recently, large language models (LLMs) shows the powerful ability to integrate various NLP tasks with the chat paradigm. In this way, this paper proposes a new Chat-based Aspect Sentiment Understanding (ChatASU) task, aiming to explore LLMs' ability in understanding aspect sentiments in dialogue scenarios. Particularly, this ChatASU task introduces a sub-task, i.e., Aspect Chain Reasoning (ACR) task, to address the aspect coreference issue. On this basis, we propose a Trusted Self-reflexion Approach (TSA) with ChatGLM as backbone to ChatASU. Specifically, this TSA treats the ACR task as an auxiliary task to boost the performance of the primary ASU task, and further integrates trusted learning into reflexion mechanisms to alleviate the LLMs-intrinsic factual hallucination problem in TSA. Furthermore, a high-quality ChatASU dataset is annotated to evaluate TSA, and extensive experiments show that our proposed TSA can significantly outperform several state-of-the-art baselines, justifying the effectiveness of TSA to ChatASU and the importance of considering the coreference and hallucination issues in ChatASU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06921v1">GoEX: Perspectives and Designs Towards a Runtime for Autonomous LLM Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are evolving beyond their classical role of providing information within dialogue systems to actively engaging with tools and performing actions on real-world applications and services. Today, humans verify the correctness and appropriateness of the LLM-generated outputs (e.g., code, functions, or actions) before putting them into real-world execution. This poses significant challenges as code comprehension is well known to be notoriously difficult. In this paper, we study how humans can efficiently collaborate with, delegate to, and supervise autonomous LLMs in the future. We argue that in many cases, "post-facto validation" - verifying the correctness of a proposed action after seeing the output - is much easier than the aforementioned "pre-facto validation" setting. The core concept behind enabling a post-facto validation system is the integration of an intuitive undo feature, and establishing a damage confinement for the LLM-generated actions as effective strategies to mitigate the associated risks. Using this, a human can now either revert the effect of an LLM-generated output or be confident that the potential risk is bounded. We believe this is critical to unlock the potential for LLM agents to interact with applications and services with limited (post-facto) human involvement. We describe the design and implementation of our open-source runtime for executing LLM actions, Gorilla Execution Engine (GoEX), and present open research questions towards realizing the goal of LLMs and applications interacting with each other with minimal human supervision. We release GoEX at https://github.com/ShishirPatil/gorilla/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06480v2">Ada-LEval: Evaluating long-context LLMs with length-adaptable benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-04-10
      | 💬 NAACL 2024
    </div>
    <details class="paper-abstract">
      Recently, the large language model (LLM) community has shown increasing interest in enhancing LLMs' capability to handle extremely long documents. As various long-text techniques and model architectures emerge, the precise and detailed evaluation of models' long-text capabilities has become increasingly important. Existing long-text evaluation benchmarks, such as L-Eval and LongBench, construct long-text test sets based on open-source datasets, focusing mainly on QA and summarization tasks. These datasets include test samples of varying lengths (from 2k to 32k+) entangled together, making it challenging to assess model capabilities across different length ranges. Moreover, they do not cover the ultralong settings (100k+ tokens) that the latest LLMs claim to achieve. In this paper, we introduce Ada-LEval, a length-adaptable benchmark for evaluating the long-context understanding of LLMs. Ada-LEval includes two challenging subsets, TSort and BestAnswer, which enable a more reliable evaluation of LLMs' long context capabilities. These benchmarks support intricate manipulation of the length of test cases, and can easily produce text samples up to 128k tokens. We evaluate 4 state-of-the-art closed-source API models and 6 open-source models with Ada-LEval. The evaluation results demonstrate the limitations of current LLMs, especially in ultra-long-context settings. Our code is available at https://github.com/open-compass/Ada-LEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05125v3">Zero-Shot Clinical Trial Patient Matching with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-10
    </div>
    <details class="paper-abstract">
      Matching patients to clinical trials is a key unsolved challenge in bringing new drugs to market. Today, identifying patients who meet a trial's eligibility criteria is highly manual, taking up to 1 hour per patient. Automated screening is challenging, however, as it requires understanding unstructured clinical text. Large language models (LLMs) offer a promising solution. In this work, we explore their application to trial matching. First, we design an LLM-based system which, given a patient's medical history as unstructured clinical text, evaluates whether that patient meets a set of inclusion criteria (also specified as free text). Our zero-shot system achieves state-of-the-art scores on the n2c2 2018 cohort selection benchmark. Second, we improve the data and cost efficiency of our method by identifying a prompting strategy which matches patients an order of magnitude faster and more cheaply than the status quo, and develop a two-stage retrieval pipeline that reduces the number of tokens processed by up to a third while retaining high performance. Third, we evaluate the interpretability of our system by having clinicians evaluate the natural language justifications generated by the LLM for each eligibility decision, and show that it can output coherent explanations for 97% of its correct decisions and 75% of its incorrect ones. Our results establish the feasibility of using LLMs to accelerate clinical trial operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06664v1">CulturalTeaming: AI-Assisted Interactive Red-Teaming for Challenging LLMs' (Lack of) Multicultural Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-04-10
      | 💬 Preprint (under review)
    </div>
    <details class="paper-abstract">
      Frontier large language models (LLMs) are developed by researchers and practitioners with skewed cultural backgrounds and on datasets with skewed sources. However, LLMs' (lack of) multicultural knowledge cannot be effectively assessed with current methods for developing benchmarks. Existing multicultural evaluations primarily rely on expensive and restricted human annotations or potentially outdated internet resources. Thus, they struggle to capture the intricacy, dynamics, and diversity of cultural norms. LLM-generated benchmarks are promising, yet risk propagating the same biases they are meant to measure. To synergize the creativity and expert cultural knowledge of human annotators and the scalability and standardizability of LLM-based automation, we introduce CulturalTeaming, an interactive red-teaming system that leverages human-AI collaboration to build truly challenging evaluation dataset for assessing the multicultural knowledge of LLMs, while improving annotators' capabilities and experiences. Our study reveals that CulturalTeaming's various modes of AI assistance support annotators in creating cultural questions, that modern LLMs fail at, in a gamified manner. Importantly, the increased level of AI assistance (e.g., LLM-generated revision hints) empowers users to create more difficult questions with enhanced perceived creativity of themselves, shedding light on the promises of involving heavier AI assistance in modern evaluation dataset creation procedures. Through a series of 1-hour workshop sessions, we gather CULTURALBENCH-V0.1, a compact yet high-quality evaluation dataset with users' red-teaming attempts, that different families of modern LLMs perform with accuracy ranging from 37.7% to 72.2%, revealing a notable gap in LLMs' multicultural proficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06644v1">Khayyam Challenge (PersianMMLU): Is Your LLM Truly Wise to The Persian Language?</a></div>
    <div class="paper-meta">
      📅 2024-04-09
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) is challenging due to their generative nature, necessitating precise evaluation methodologies. Additionally, non-English LLM evaluation lags behind English, resulting in the absence or weakness of LLMs for many languages. In response to this necessity, we introduce Khayyam Challenge (also known as PersianMMLU), a meticulously curated collection comprising 20,192 four-choice questions sourced from 38 diverse tasks extracted from Persian examinations, spanning a wide spectrum of subjects, complexities, and ages. The primary objective of the Khayyam Challenge is to facilitate the rigorous evaluation of LLMs that support the Persian language. Distinctive features of the Khayyam Challenge are (i) its comprehensive coverage of various topics, including literary comprehension, mathematics, sciences, logic, intelligence testing, etc., aimed at assessing different facets of LLMs such as language comprehension, reasoning, and information retrieval across various educational stages, from lower primary school to upper secondary school (ii) its inclusion of rich metadata such as human response rates, difficulty levels, and descriptive answers (iii) its utilization of new data to avoid data contamination issues prevalent in existing frameworks (iv) its use of original, non-translated data tailored for Persian speakers, ensuring the framework is free from translation challenges and errors while encompassing cultural nuances (v) its inherent scalability for future data updates and evaluations without requiring special human effort. Previous works lacked an evaluation framework that combined all of these features into a single comprehensive benchmark. Furthermore, we evaluate a wide range of existing LLMs that support the Persian language, with statistical analyses and interpretations of their outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07242v1">Sandwich attack: Multi-language Mixture Adaptive Attack on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being developed and applied, but their widespread use faces challenges. These include aligning LLMs' responses with human values to prevent harmful outputs, which is addressed through safety training methods. Even so, bad actors and malicious users have succeeded in attempts to manipulate the LLMs to generate misaligned responses for harmful questions such as methods to create a bomb in school labs, recipes for harmful drugs, and ways to evade privacy rights. Another challenge is the multilingual capabilities of LLMs, which enable the model to understand and respond in multiple languages. Consequently, attackers exploit the unbalanced pre-training datasets of LLMs in different languages and the comparatively lower model performance in low-resource languages than high-resource ones. As a result, attackers use a low-resource languages to intentionally manipulate the model to create harmful responses. Many of the similar attack vectors have been patched by model providers, making the LLMs more robust against language-based manipulation. In this paper, we introduce a new black-box attack vector called the \emph{Sandwich attack}: a multi-language mixture attack, which manipulates state-of-the-art LLMs into generating harmful and misaligned responses. Our experiments with five different models, namely Google's Bard, Gemini Pro, LLaMA-2-70-B-Chat, GPT-3.5-Turbo, GPT-4, and Claude-3-OPUS, show that this attack vector can be used by adversaries to generate harmful responses and elicit misaligned responses from these models. By detailing both the mechanism and impact of the Sandwich attack, this paper aims to guide future research and development towards more secure and resilient LLMs, ensuring they serve the public good while minimizing potential for misuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06503v1">Comparing Two Model Designs for Clinical Note Generation; Is an LLM a Useful Evaluator of Consistency?</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 Accepted to NAACL 2024 Findings
    </div>
    <details class="paper-abstract">
      Following an interaction with a patient, physicians are responsible for the submission of clinical documentation, often organized as a SOAP note. A clinical note is not simply a summary of the conversation but requires the use of appropriate medical terminology. The relevant information can then be extracted and organized according to the structure of the SOAP note. In this paper we analyze two different approaches to generate the different sections of a SOAP note based on the audio recording of the conversation, and specifically examine them in terms of note consistency. The first approach generates the sections independently, while the second method generates them all together. In this work we make use of PEGASUS-X Transformer models and observe that both methods lead to similar ROUGE values (less than 1% difference) and have no difference in terms of the Factuality metric. We perform a human evaluation to measure aspects of consistency and demonstrate that LLMs like Llama2 can be used to perform the same tasks with roughly the same agreement as the human annotators. Between the Llama2 analysis and the human reviewers we observe a Cohen Kappa inter-rater reliability of 0.79, 1.00, and 0.32 for consistency of age, gender, and body part injury, respectively. With this we demonstrate the usefulness of leveraging an LLM to measure quality indicators that can be identified by humans but are not currently captured by automatic metrics. This allows scaling evaluation to larger data sets, and we find that clinical note consistency improves by generating each new section conditioned on the output of all previously generated sections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06488v1">Pitfalls of Conversational LLMs on News Debiasing</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 The paper is accepted at the DELITE workshop which is co-located at COLING/LREC
    </div>
    <details class="paper-abstract">
      This paper addresses debiasing in news editing and evaluates the effectiveness of conversational Large Language Models in this task. We designed an evaluation checklist tailored to news editors' perspectives, obtained generated texts from three popular conversational models using a subset of a publicly available dataset in media bias, and evaluated the texts according to the designed checklist. Furthermore, we examined the models as evaluator for checking the quality of debiased model outputs. Our findings indicate that none of the LLMs are perfect in debiasing. Notably, some models, including ChatGPT, introduced unnecessary changes that may impact the author's style and create misinformation. Lastly, we show that the models do not perform as proficiently as domain experts in evaluating the quality of debiased outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06411v1">AgentQuest: A Modular Benchmark Framework to Measure Progress and Improve LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 Accepted at the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT 2024)
    </div>
    <details class="paper-abstract">
      The advances made by Large Language Models (LLMs) have led to the pursuit of LLM agents that can solve intricate, multi-step reasoning tasks. As with any research pursuit, benchmarking and evaluation are key corner stones to efficient and reliable progress. However, existing benchmarks are often narrow and simply compute overall task success. To face these issues, we propose AgentQuest -- a framework where (i) both benchmarks and metrics are modular and easily extensible through well documented and easy-to-use APIs; (ii) we offer two new evaluation metrics that can reliably track LLM agent progress while solving a task. We exemplify the utility of the metrics on two use cases wherein we identify common failure points and refine the agent architecture to obtain a significant performance increase. Together with the research community, we hope to extend AgentQuest further and therefore we make it available under https://github.com/nec-research/agentquest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06283v1">LLMs' Reading Comprehension Is Affected by Parametric Knowledge and Struggles with Hypothetical Statements</a></div>
    <div class="paper-meta">
      📅 2024-04-09
    </div>
    <details class="paper-abstract">
      The task of reading comprehension (RC), often implemented as context-based question answering (QA), provides a primary means to assess language models' natural language understanding (NLU) capabilities. Yet, when applied to large language models (LLMs) with extensive built-in world knowledge, this method can be deceptive. If the context aligns with the LLMs' internal knowledge, it is hard to discern whether the models' answers stem from context comprehension or from LLMs' internal information. Conversely, using data that conflicts with the models' knowledge creates erroneous trends which distort the results. To address this issue, we suggest to use RC on imaginary data, based on fictitious facts and entities. This task is entirely independent of the models' world knowledge, enabling us to evaluate LLMs' linguistic abilities without the interference of parametric knowledge. Testing ChatGPT, GPT-4, LLaMA 2 and Mixtral on such imaginary data, we uncover a class of linguistic phenomena posing a challenge to current LLMs, involving thinking in terms of alternative, hypothetical scenarios. While all the models handle simple affirmative and negative contexts with high accuracy, they are much more prone to error when dealing with modal and conditional contexts. Crucially, these phenomena also trigger the LLMs' vulnerability to knowledge-conflicts again. In particular, while some models prove virtually unaffected by knowledge conflicts in affirmative and negative contexts, when faced with more semantically involved modal and conditional environments, they often fail to separate the text from their internal knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11858v1">Student Perspectives on Using a Large Language Model (LLM) for an Assignment on Professional Ethics</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 accepted at ITiCSE 2024, Milan, Italy
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) started a serious discussion among educators on how LLMs would affect, e.g., curricula, assessments, and students' competencies. Generative AI and LLMs also raised ethical questions and concerns for computing educators and professionals. This experience report presents an assignment within a course on professional competencies, including some related to ethics, that computing master's students need in their careers. For the assignment, student groups discussed the ethical process by Lennerfors et al. by analyzing a case: a fictional researcher considers whether to attend the real CHI 2024 conference in Hawaii. The tasks were (1) to participate in in-class discussions on the case, (2) to use an LLM of their choice as a discussion partner for said case, and (3) to document both discussions, reflecting on their use of the LLM. Students reported positive experiences with the LLM as a way to increase their knowledge and understanding, although some identified limitations. The LLM provided a wider set of options for action in the studied case, including unfeasible ones. The LLM would not select a course of action, so students had to choose themselves, which they saw as coherent. From the educators' perspective, there is a need for more instruction for students using LLMs: some students did not perceive the tools as such but rather as an authoritative knowledge base. Therefore, this work has implications for educators considering the use of LLMs as discussion partners or tools to practice critical thinking, especially in computing ethics education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12869v2">Exploring the Impact of Table-to-Text Methods on Augmenting LLM-based Question Answering with Domain Hybrid Data</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 Accepted to NAACL 2024 Industry Track Paper
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) for Question Answering (QA) with domain specific data has attracted wide attention. However, domain data often exists in a hybrid format, including text and semi-structured tables, posing challenges for the seamless integration of information. Table-to-Text Generation is a promising solution by facilitating the transformation of hybrid data into a uniformly text-formatted corpus. Although this technique has been widely studied by the NLP community, there is currently no comparative analysis on how corpora generated by different table-to-text methods affect the performance of QA systems. In this paper, we address this research gap in two steps. First, we innovatively integrate table-to-text generation into the framework of enhancing LLM-based QA systems with domain hybrid data. Then, we utilize this framework in real-world industrial data to conduct extensive experiments on two types of QA systems (DSFT and RAG frameworks) with four representative methods: Markdown format, Template serialization, TPLM-based method, and LLM-based method. Based on the experimental results, we draw some empirical findings and explore the underlying reasons behind the success of some methods. We hope the findings of this work will provide a valuable reference for the academic and industrial communities in developing robust QA systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06082v1">A RAG Method for Source Code Inquiry Tailored to Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 6 pages, 2 columns, English translation of the manuscript originally presented in Japanese at a domestic workshop
    </div>
    <details class="paper-abstract">
      Although the context length limitation of large language models (LLMs) has been mitigated, it still hinders their application to software development tasks. This study proposes a method incorporating execution traces into RAG for inquiries about source code. Small-scale experiments confirm a tendency for the method to contribute to improving LLM response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06041v1">On Evaluating the Efficiency of Source Code Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-09
      | 💬 1st special event of AI Foundation Models and Software Engineering (FORGE 2024)
    </div>
    <details class="paper-abstract">
      Recent years have seen the remarkable capabilities of large language models (LLMs) for code generation. Different from existing work that evaluate the correctness of the code generated by LLMs, we propose to further evaluate its efficiency. More efficient code can lead to higher performance and execution efficiency of programs and software completed by LLM-assisted programming. First, we evaluate the efficiency of the code generated by LLMs on two benchmarks, HumanEval and MBPP. Then, we choose a set of programming problems from the online judge platform LeetCode to conduct a more difficult evaluation. Finally, we explore several prompts that would enable LLMs to generate more efficient code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.06035v1">PM4Py.LLM: a Comprehensive Module for Implementing PM on LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-09
    </div>
    <details class="paper-abstract">
      pm4py is a process mining library for Python implementing several process mining (PM) artifacts and algorithms. It also offers methods to integrate PM with large language models (LLMs). This paper examines how the current paradigms of PM on LLM are implemented in pm4py, identifying challenges such as privacy, hallucinations, and the context window limit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2404.05213v1">Evaluation of an LLM in Identifying Logical Fallacies: A Call for Rigor When Adopting LLMs in HCI Research</a></div>
    <div class="paper-meta">
      📅 2024-04-09
    </div>
    <details class="paper-abstract">
      There is increasing interest in the adoption of LLMs in HCI research. However, LLMs may often be regarded as a panacea because of their powerful capabilities with an accompanying oversight on whether they are suitable for their intended tasks. We contend that LLMs should be adopted in a critical manner following rigorous evaluation. Accordingly, we present the evaluation of an LLM in identifying logical fallacies that will form part of a digital misinformation intervention. By comparing to a labeled dataset, we found that GPT-4 achieves an accuracy of 0.79, and for our intended use case that excludes invalid or unidentified instances, an accuracy of 0.90. This gives us the confidence to proceed with the application of the LLM while keeping in mind the areas where it still falls short. The paper describes our evaluation approach, results and reflections on the use of the LLM for our intended task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18249v2">Exploring the Deceptive Power of LLM-Generated Fake News: A Study of Real-World Detection Challenges</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have enabled the creation of fake news, particularly in complex fields like healthcare. Studies highlight the gap in the deceptive power of LLM-generated fake news with and without human assistance, yet the potential of prompting techniques has not been fully explored. Thus, this work aims to determine whether prompting strategies can effectively narrow this gap. Current LLM-based fake news attacks require human intervention for information gathering and often miss details and fail to maintain context consistency. Therefore, to better understand threat tactics, we propose a strong fake news attack method called conditional Variational-autoencoder-Like Prompt (VLPrompt). Unlike current methods, VLPrompt eliminates the need for additional data collection while maintaining contextual coherence and preserving the intricacies of the original text. To propel future research on detecting VLPrompt attacks, we created a new dataset named VLPrompt fake news (VLPFN) containing real and fake texts. Our experiments, including various detection methods and novel human study metrics, were conducted to assess their performance on our dataset, yielding numerous findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05825v1">LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      Recently embedding-based retrieval or dense retrieval have shown state of the art results, compared with traditional sparse or bag-of-words based approaches. This paper introduces a model-agnostic doc-level embedding framework through large language model (LLM) augmentation. In addition, it also improves some important components in the retrieval model training process, such as negative sampling, loss function, etc. By implementing this LLM-augmented retrieval framework, we have been able to significantly improve the effectiveness of widely-used retriever models such as Bi-encoders (Contriever, DRAGON) and late-interaction models (ColBERTv2), thereby achieving state-of-the-art results on LoTTE datasets and BEIR datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05719v1">Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have been noteworthy, yet, these general-domain MLLMs often fall short in their ability to comprehend and interact effectively with user interface (UI) screens. In this paper, we present Ferret-UI, a new MLLM tailored for enhanced understanding of mobile UI screens, equipped with referring, grounding, and reasoning capabilities. Given that UI screens typically exhibit a more elongated aspect ratio and contain smaller objects of interest (e.g., icons, texts) than natural images, we incorporate "any resolution" on top of Ferret to magnify details and leverage enhanced visual features. Specifically, each screen is divided into 2 sub-images based on the original aspect ratio (i.e., horizontal division for portrait screens and vertical division for landscape screens). Both sub-images are encoded separately before being sent to LLMs. We meticulously gather training samples from an extensive range of elementary UI tasks, such as icon recognition, find text, and widget listing. These samples are formatted for instruction-following with region annotations to facilitate precise referring and grounding. To augment the model's reasoning ability, we further compile a dataset for advanced tasks, including detailed description, perception/interaction conversations, and function inference. After training on the curated datasets, Ferret-UI exhibits outstanding comprehension of UI screens and the capability to execute open-ended instructions. For model evaluation, we establish a comprehensive benchmark encompassing all the aforementioned tasks. Ferret-UI excels not only beyond most open-source UI MLLMs, but also surpasses GPT-4V on all the elementary UI tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12089v2">HiCRISP: An LLM-based Hierarchical Closed-Loop Robotic Intelligent Self-Correction Planner</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into robotics has revolutionized human-robot interactions and autonomous task planning. However, these systems are often unable to self-correct during the task execution, which hinders their adaptability in dynamic real-world environments. To address this issue, we present a Hierarchical Closed-loop Robotic Intelligent Self-correction Planner (HiCRISP), an innovative framework that enables robots to correct errors within individual steps during the task execution. HiCRISP actively monitors and adapts the task execution process, addressing both high-level planning and low-level action errors. Extensive benchmark experiments, encompassing virtual and real-world scenarios, showcase HiCRISP's exceptional performance, positioning it as a promising solution for robotic task planning with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08001v1">Xiwu: A Basis Flexible and Learnable LLM for High Energy Physics</a></div>
    <div class="paper-meta">
      📅 2024-04-08
      | 💬 15 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are undergoing a period of rapid updates and changes, with state-of-the-art (SOTA) model frequently being replaced. When applying LLMs to a specific scientific field, it's challenging to acquire unique domain knowledge while keeping the model itself advanced. To address this challenge, a sophisticated large language model system named as Xiwu has been developed, allowing you switch between the most advanced foundation models and quickly teach the model domain knowledge. In this work, we will report on the best practices for applying LLMs in the field of high-energy physics (HEP), including: a seed fission technology is proposed and some data collection and cleaning tools are developed to quickly obtain domain AI-Ready dataset; a just-in-time learning system is implemented based on the vector store technology; an on-the-fly fine-tuning system has been developed to facilitate rapid training under a specified foundation model. The results show that Xiwu can smoothly switch between foundation models such as LLaMA, Vicuna, ChatGLM and Grok-1. The trained Xiwu model is significantly outperformed the benchmark model on the HEP knowledge question-and-answering and code generation. This strategy significantly enhances the potential for growth of our model's performance, with the hope of surpassing GPT-4 as it evolves with the development of open-source models. This work provides a customized LLM for the field of HEP, while also offering references for applying LLM to other fields, the corresponding codes are available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05213v1">Evaluation of an LLM in Identifying Logical Fallacies: A Call for Rigor When Adopting LLMs in HCI Research</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      There is increasing interest in the adoption of LLMs in HCI research. However, LLMs may often be regarded as a panacea because of their powerful capabilities with an accompanying oversight on whether they are suitable for their intended tasks. We contend that LLMs should be adopted in a critical manner following rigorous evaluation. Accordingly, we present the evaluation of an LLM in identifying logical fallacies that will form part of a digital misinformation intervention. By comparing to a labeled dataset, we found that GPT-4 achieves an accuracy of 0.79, and for our intended use case that excludes invalid or unidentified instances, an accuracy of 0.90. This gives us the confidence to proceed with the application of the LLM while keeping in mind the areas where it still falls short. The paper describes our evaluation approach, results and reflections on the use of the LLM for our intended task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05183v1">Progressive Alignment with VLM-LLM Feature to Augment Defect Classification for the ASE Dataset</a></div>
    <div class="paper-meta">
      📅 2024-04-08
      | 💬 MULA 2024
    </div>
    <details class="paper-abstract">
      Traditional defect classification approaches are facing with two barriers. (1) Insufficient training data and unstable data quality. Collecting sufficient defective sample is expensive and time-costing, consequently leading to dataset variance. It introduces the difficulty on recognition and learning. (2) Over-dependence on visual modality. When the image pattern and texture is monotonic for all defect classes in a given dataset, the performance of conventional AOI system cannot be guaranteed. In scenarios where image quality is compromised due to mechanical failures or when defect information is inherently difficult to discern, the performance of deep models cannot be guaranteed. A main question is, "how to solve those two problems when they occur at the same time?" The feasible strategy is to explore another feature within dataset and combine an eminent vision-language model (VLM) and Large-Language model (LLM) with their astonishing zero-shot capability. In this work, we propose the special ASE dataset, including rich data description recorded on image, for defect classification, but the defect feature is uneasy to learn directly. Secondly, We present the prompting for VLM-LLM against defect classification with the proposed ASE dataset to activate extra-modality feature from images to enhance performance. Then, We design the novel progressive feature alignment (PFA) block to refine image-text feature to alleviate the difficulty of alignment under few-shot scenario. Finally, the proposed Cross-modality attention fusion (CMAF) module can effectively fuse different modality feature. Experiment results have demonstrated our method's effectiveness over several defect classification methods for the ASE dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12574v2">Modeling Uncertainty and Using Post-fusion as Fallback Improves Retrieval Augmented Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-08
    </div>
    <details class="paper-abstract">
      The integration of retrieved passages and large language models (LLMs), such as ChatGPTs, has significantly contributed to improving open-domain question answering. However, there is still a lack of exploration regarding the optimal approach for incorporating retrieved passages into the answer generation process. This paper aims to fill this gap by investigating different methods of combining retrieved passages with LLMs to enhance answer generation. We begin by examining the limitations of a commonly-used concatenation approach. Surprisingly, this approach often results in generating "unknown" outputs, even when the correct document is among the top-k retrieved passages. To address this issue, we explore four alternative strategies for integrating the retrieved passages with the LLMs. These strategies include two single-round methods that utilize chain-of-thought reasoning and two multi-round strategies that incorporate feedback loops. Through comprehensive analyses and experiments, we provide insightful observations on how to effectively leverage retrieved passages to enhance the answer generation capability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.05144v1">Enhancing Clinical Efficiency through LLM: Discharge Note Generation for Cardiac Patients</a></div>
    <div class="paper-meta">
      📅 2024-04-08
      | 💬 10 pages, 1 figure, 3 tables, conference
    </div>
    <details class="paper-abstract">
      Medical documentation, including discharge notes, is crucial for ensuring patient care quality, continuity, and effective medical communication. However, the manual creation of these documents is not only time-consuming but also prone to inconsistencies and potential errors. The automation of this documentation process using artificial intelligence (AI) represents a promising area of innovation in healthcare. This study directly addresses the inefficiencies and inaccuracies in creating discharge notes manually, particularly for cardiac patients, by employing AI techniques, specifically large language model (LLM). Utilizing a substantial dataset from a cardiology center, encompassing wide-ranging medical records and physician assessments, our research evaluates the capability of LLM to enhance the documentation process. Among the various models assessed, Mistral-7B distinguished itself by accurately generating discharge notes that significantly improve both documentation efficiency and the continuity of care for patients. These notes underwent rigorous qualitative evaluation by medical expert, receiving high marks for their clinical relevance, completeness, readability, and contribution to informed decision-making and care planning. Coupled with quantitative analyses, these results confirm Mistral-7B's efficacy in distilling complex medical information into concise, coherent summaries. Overall, our findings illuminate the considerable promise of specialized LLM, such as Mistral-7B, in refining healthcare documentation workflows and advancing patient care. This study lays the groundwork for further integrating advanced AI technologies in healthcare, demonstrating their potential to revolutionize patient documentation and support better care outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.09203v2">Measurement in the Age of LLMs: An Application to Ideological Scaling</a></div>
    <div class="paper-meta">
      📅 2024-04-08
      | 💬 Under review a Harvard Data Science Review. Previously presented at the 4th International Conference of Social Computing in Beijing, China, September 2023, the New Directions in Analyzing Text as Data (TADA) meeting in Amherst, MA, USA, November 2023, and the NeurIPS workshop titled "I Can't Believe It's Not Better!'' Failure Modes in the Age of Foundation Models in New Orleans, LA, December 2023
    </div>
    <details class="paper-abstract">
      Much of social science is centered around terms like ``ideology'' or ``power'', which generally elude precise definition, and whose contextual meanings are trapped in surrounding language. This paper explores the use of large language models (LLMs) to flexibly navigate the conceptual clutter inherent to social scientific measurement tasks. We rely on LLMs' remarkable linguistic fluency to elicit ideological scales of both legislators and text, which accord closely to established methods and our own judgement. A key aspect of our approach is that we elicit such scores directly, instructing the LLM to furnish numeric scores itself. This approach affords a great deal of flexibility, which we showcase through a variety of different case studies. Our results suggest that LLMs can be used to characterize highly subtle and diffuse manifestations of political ideology in text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04966v1">Enhancing LLM-based Test Generation for Hard-to-Cover Branches via Program Analysis</a></div>
    <div class="paper-meta">
      📅 2024-04-07
      | 💬 11 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Automatic test generation plays a critical role in software quality assurance. While the recent advances in Search-Based Software Testing (SBST) and Large Language Models (LLMs) have shown promise in generating useful tests, these techniques still struggle to cover certain branches. Reaching these hard-to-cover branches usually requires constructing complex objects and resolving intricate inter-procedural dependencies in branch conditions, which poses significant challenges for existing test generation techniques. In this work, we propose TELPA, a novel technique aimed at addressing these challenges. Its key insight lies in extracting real usage scenarios of the target method under test to learn how to construct complex objects and extracting methods entailing inter-procedural dependencies with hard-to-cover branches to learn the semantics of branch constraints. To enhance efficiency and effectiveness, TELPA identifies a set of ineffective tests as counter-examples for LLMs and employs a feedback-based process to iteratively refine these counter-examples. Then, TELPA integrates program analysis results and counter-examples into the prompt, guiding LLMs to gain deeper understandings of the semantics of the target method and generate diverse tests that can reach the hard-to-cover branches. Our experimental results on 27 open-source Python projects demonstrate that TELPA significantly outperforms the state-of-the-art SBST and LLM-based techniques, achieving an average improvement of 31.39% and 22.22% in terms of branch coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04902v1">AI2Apps: A Visual IDE for Building LLM-based AI Agent Applications</a></div>
    <div class="paper-meta">
      📅 2024-04-07
    </div>
    <details class="paper-abstract">
      We introduce AI2Apps, a Visual Integrated Development Environment (Visual IDE) with full-cycle capabilities that accelerates developers to build deployable LLM-based AI agent Applications. This Visual IDE prioritizes both the Integrity of its development tools and the Visuality of its components, ensuring a smooth and efficient building experience.On one hand, AI2Apps integrates a comprehensive development toolkit ranging from a prototyping canvas and AI-assisted code editor to agent debugger, management system, and deployment tools all within a web-based graphical user interface. On the other hand, AI2Apps visualizes reusable front-end and back-end code as intuitive drag-and-drop components. Furthermore, a plugin system named AI2Apps Extension (AAE) is designed for Extensibility, showcasing how a new plugin with 20 components enables web agent to mimic human-like browsing behavior. Our case study demonstrates substantial efficiency improvements, with AI2Apps reducing token consumption and API calls when debugging a specific sophisticated multimodal agent by approximately 90% and 80%, respectively. The AI2Apps, including an online demo, open-source code, and a screencast video, is now publicly accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04809v1">Low-Resource Machine Translation through Retrieval-Augmented LLM Prompting: A Study on the Mambai Language</a></div>
    <div class="paper-meta">
      📅 2024-04-07
    </div>
    <details class="paper-abstract">
      This study explores the use of large language models (LLMs) for translating English into Mambai, a low-resource Austronesian language spoken in Timor-Leste, with approximately 200,000 native speakers. Leveraging a novel corpus derived from a Mambai language manual and additional sentences translated by a native speaker, we examine the efficacy of few-shot LLM prompting for machine translation (MT) in this low-resource context. Our methodology involves the strategic selection of parallel sentences and dictionary entries for prompting, aiming to enhance translation accuracy, using open-source and proprietary LLMs (LlaMa 2 70b, Mixtral 8x7B, GPT-4). We find that including dictionary entries in prompts and a mix of sentences retrieved through TF-IDF and semantic embeddings significantly improves translation quality. However, our findings reveal stark disparities in translation performance across test sets, with BLEU scores reaching as high as 21.2 on materials from the language manual, in contrast to a maximum of 4.4 on a test set provided by a native speaker. These results underscore the importance of diverse and representative corpora in assessing MT for low-resource languages. Our research provides insights into few-shot LLM prompting for low-resource MT, and makes available an initial corpus for the Mambai language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04689v1">Multicalibration for Confidence Scoring in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-06
    </div>
    <details class="paper-abstract">
      This paper proposes the use of "multicalibration" to yield interpretable and reliable confidence scores for outputs generated by large language models (LLMs). Multicalibration asks for calibration not just marginally, but simultaneously across various intersecting groupings of the data. We show how to form groupings for prompt/completion pairs that are correlated with the probability of correctness via two techniques: clustering within an embedding space, and "self-annotation" - querying the LLM by asking it various yes-or-no questions about the prompt. We also develop novel variants of multicalibration algorithms that offer performance improvements by reducing their tendency to overfit. Through systematic benchmarking across various question answering datasets and LLMs, we show how our techniques can yield confidence scores that provide substantial improvements in fine-grained measures of both calibration and accuracy compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04631v1">On the Limitations of Large Language Models (LLMs): False Attribution</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      In this work, we provide insight into one important limitation of large language models (LLMs), i.e. false attribution, and introduce a new hallucination metric - Simple Hallucination Index (SHI). The task of automatic author attribution for relatively small chunks of text is an important NLP task but can be challenging. We empirically evaluate the power of 3 open SotA LLMs in zero-shot setting (LLaMA-2-13B, Mixtral 8x7B, and Gemma-7B), especially as human annotation can be costly. We collected the top 10 most popular books, according to Project Gutenberg, divided each one into equal chunks of 400 words, and asked each LLM to predict the author. We then randomly sampled 162 chunks for human evaluation from each of the annotated books, based on the error margin of 7% and a confidence level of 95% for the book with the most chunks (Great Expectations by Charles Dickens, having 922 chunks). The average results show that Mixtral 8x7B has the highest prediction accuracy, the lowest SHI, and a Pearson's correlation (r) of 0.737, 0.249, and -0.9996, respectively, followed by LLaMA-2-13B and Gemma-7B. However, Mixtral 8x7B suffers from high hallucinations for 3 books, rising as high as an SHI of 0.87 (in the range 0-1, where 1 is the worst). The strong negative correlation of accuracy and SHI, given by r, demonstrates the fidelity of the new hallucination metric, which is generalizable to other tasks. We publicly release the annotated chunks of data and our codes to aid the reproducibility and evaluation of other models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04570v1">A Map of Exploring Human Interaction patterns with LLM: Insights into Collaboration and Creativity</a></div>
    <div class="paper-meta">
      📅 2024-04-06
    </div>
    <details class="paper-abstract">
      The outstanding performance capabilities of large language model have driven the evolution of current AI system interaction patterns. This has led to considerable discussion within the Human-AI Interaction (HAII) community. Numerous studies explore this interaction from technical, design, and empirical perspectives. However, the majority of current literature reviews concentrate on interactions across the wider spectrum of AI, with limited attention given to the specific realm of interaction with LLM. We searched for articles on human interaction with LLM, selecting 110 relevant publications meeting consensus definition of Human-AI interaction. Subsequently, we developed a comprehensive Mapping Procedure, structured in five distinct stages, to systematically analyze and categorize the collected publications. Applying this methodical approach, we meticulously mapped the chosen studies, culminating in a detailed and insightful representation of the research landscape. Overall, our review presents an novel approach, introducing a distinctive mapping method, specifically tailored to evaluate human-LLM interaction patterns. We conducted a comprehensive analysis of the current research in related fields, employing clustering techniques for categorization, which enabled us to clearly delineate the status and challenges prevalent in each identified area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06652v1">Large Language Model (LLM) AI text generation detection based on transformer deep learning algorithm</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      In this paper, a tool for detecting LLM AI text generation is developed based on the Transformer model, aiming to improve the accuracy of AI text generation detection and provide reference for subsequent research. Firstly the text is Unicode normalised, converted to lowercase form, characters other than non-alphabetic characters and punctuation marks are removed by regular expressions, spaces are added around punctuation marks, first and last spaces are removed, consecutive ellipses are replaced with single spaces and the text is connected using the specified delimiter. Next remove non-alphabetic characters and extra whitespace characters, replace multiple consecutive whitespace characters with a single space and again convert to lowercase form. The deep learning model combines layers such as LSTM, Transformer and CNN for text classification or sequence labelling tasks. The training and validation sets show that the model loss decreases from 0.127 to 0.005 and accuracy increases from 94.96 to 99.8, indicating that the model has good detection and classification ability for AI generated text. The test set confusion matrix and accuracy show that the model has 99% prediction accuracy for AI-generated text, with a precision of 0.99, a recall of 1, and an f1 score of 0.99, achieving a very high classification accuracy. Looking forward, it has the prospect of wide application in the field of AI text detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.06875v4">nanoLM: an Affordable LLM Pre-training Benchmark via Accurate Loss Prediction across Scales</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 This is a modified and extended version of our previous Mu-scaling work released in April 2023 (see v1)
    </div>
    <details class="paper-abstract">
      As language models scale up, it becomes increasingly expensive to verify research ideas because conclusions on small models do not trivially transfer to large ones. A possible solution is to establish a generic system that accurately predicts certain metrics for large models without training them. Existing scaling laws require hyperparameter search on the largest models, limiting their predicative capability. In this paper, we present an approach (namely {\mu}Scaling) to predict the pre-training loss, based on our observations that Maximal Update Parametrization ({\mu}P) enables accurate fitting of scaling laws close to common loss basins in hyperparameter space. With {\mu}Scaling, different model designs can be compared on large scales by training only their smaller counterparts. Further, we introduce nanoLM: an affordable LLM pre-training benchmark that facilitates this new research paradigm. With around 14% of the one-time pre-training cost, we can accurately forecast the loss for models up to 52B. Our goal with nanoLM is to empower researchers with limited resources to reach meaningful conclusions on large models. We also aspire for our benchmark to serve as a bridge between the academic community and the industry. Code for {\mu}Scaling is available at https://github.com/cofe-ai/Mu-scaling. Code for nanoLLM will be available later.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04510v1">IITK at SemEval-2024 Task 2: Exploring the Capabilities of LLMs for Safe Biomedical Natural Language Inference for Clinical Trials</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 Accepted at SemEval 2024, NAACL 2024; 8 Pages
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have demonstrated state-of-the-art performance in various natural language processing (NLP) tasks across multiple domains, yet they are prone to shortcut learning and factual inconsistencies. This research investigates LLMs' robustness, consistency, and faithful reasoning when performing Natural Language Inference (NLI) on breast cancer Clinical Trial Reports (CTRs) in the context of SemEval 2024 Task 2: Safe Biomedical Natural Language Inference for Clinical Trials. We examine the reasoning capabilities of LLMs and their adeptness at logical problem-solving. A comparative analysis is conducted on pre-trained language models (PLMs), GPT-3.5, and Gemini Pro under zero-shot settings using Retrieval-Augmented Generation (RAG) framework, integrating various reasoning chains. The evaluation yields an F1 score of 0.69, consistency of 0.71, and a faithfulness score of 0.90 on the test dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13801v2">Natural Language as Policies: Reasoning for Coordinate-Level Embodied Control with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We demonstrate experimental results with LLMs that address robotics task planning problems. Recently, LLMs have been applied in robotics task planning, particularly using a code generation approach that converts complex high-level instructions into mid-level policy codes. In contrast, our approach acquires text descriptions of the task and scene objects, then formulates task planning through natural language reasoning, and outputs coordinate level control commands, thus reducing the necessity for intermediate representation code as policies with pre-defined APIs. Our approach is evaluated on a multi-modal prompt simulation benchmark, demonstrating that our prompt engineering experiments with natural language reasoning significantly enhance success rates compared to its absence. Furthermore, our approach illustrates the potential for natural language descriptions to transfer robotics skills from known tasks to previously unseen tasks. The project website: https://natural-language-as-policies.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03689v2">General2Specialized LLMs Translation for E-commerce</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 4 pages, 1 figure, WWW2024 accepted
    </div>
    <details class="paper-abstract">
      Existing Neural Machine Translation (NMT) models mainly handle translation in the general domain, while overlooking domains with special writing formulas, such as e-commerce and legal documents. Taking e-commerce as an example, the texts usually include amounts of domain-related words and have more grammar problems, which leads to inferior performances of current NMT methods. To address these problems, we collect two domain-related resources, including a set of term pairs (aligned Chinese-English bilingual terms) and a parallel corpus annotated for the e-commerce domain. Furthermore, we propose a two-step fine-tuning paradigm (named G2ST) with self-contrastive semantic enhancement to transfer one general NMT model to the specialized NMT model for e-commerce. The paradigm can be used for the NMT models based on Large language models (LLMs). Extensive evaluations on real e-commerce titles demonstrate the superior translation quality and robustness of our G2ST approach, as compared with state-of-the-art NMT models such as LLaMA, Qwen, GPT-3.5, and even GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.12032v5">From Quantity to Quality: Boosting LLM Performance with Self-Guided Data Selection for Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-04-06
      | 💬 NAACL, Camera-ready
    </div>
    <details class="paper-abstract">
      In the realm of Large Language Models (LLMs), the balance between instruction data quality and quantity is a focal point. Recognizing this, we introduce a self-guided methodology for LLMs to autonomously discern and select cherry samples from open-source datasets, effectively minimizing manual curation and potential cost for instruction tuning an LLM. Our key innovation, the Instruction-Following Difficulty (IFD) metric, emerges as a pivotal metric to identify discrepancies between a model's expected responses and its intrinsic generation capability. Through the application of IFD, cherry samples can be pinpointed, leading to a marked uptick in model training efficiency. Empirical validations on datasets like Alpaca and WizardLM underpin our findings; with a mere $10\%$ of original data input, our strategy showcases improved results. This synthesis of self-guided cherry-picking and the IFD metric signifies a transformative leap in the instruction tuning of LLMs, promising both efficiency and resource-conscious advancements. Codes, data, and models are available: https://github.com/tianyi-lab/Cherry_LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04376v1">ClickDiffusion: Harnessing LLMs for Interactive Precise Image Editing</a></div>
    <div class="paper-meta">
      📅 2024-04-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2402.07925
    </div>
    <details class="paper-abstract">
      Recently, researchers have proposed powerful systems for generating and manipulating images using natural language instructions. However, it is difficult to precisely specify many common classes of image transformations with text alone. For example, a user may wish to change the location and breed of a particular dog in an image with several similar dogs. This task is quite difficult with natural language alone, and would require a user to write a laboriously complex prompt that both disambiguates the target dog and describes the destination. We propose ClickDiffusion, a system for precise image manipulation and generation that combines natural language instructions with visual feedback provided by the user through a direct manipulation interface. We demonstrate that by serializing both an image and a multi-modal instruction into a textual representation it is possible to leverage LLMs to perform precise transformations of the layout and appearance of an image. Code available at https://github.com/poloclub/ClickDiffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16159v2">Designing Child-Centric AI Learning Environments: Insights from LLM-Enhanced Creative Project-Based Learning</a></div>
    <div class="paper-meta">
      📅 2024-04-05
    </div>
    <details class="paper-abstract">
      Project-based learning (PBL) is an instructional method that is very helpful in nurturing students' creativity, but it requires significant time and energy from both students and teachers. Large language models (LLMs) have been proven to assist in creative tasks, yet much controversy exists regarding their role in fostering creativity. This paper explores the potential of LLMs in PBL settings, with a special focus on fostering creativity. We began with an exploratory study involving 12 middle school students and identified five design considerations for LLM applications in PBL. Building on this, we developed an LLM-empowered, 48-hour PBL program and conducted an instructional experiment with 31 middle school students. Our results indicated that LLMs can enhance every stage of PBL. Additionally, we also discovered ambivalent perspectives among students and mentors toward LLM usage. Furthermore, we explored the challenge and design implications of integrating LLMs into PBL and reflected on the program. By bridging AI advancements into educational practice, our work aims to inspire further discourse and investigation into harnessing AI's potential in child-centric educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.10797v2">TaCo: Enhancing Cross-Lingual Transfer for Low-Resource Languages in LLMs through Translation-Assisted Chain-of-Thought Processes</a></div>
    <div class="paper-meta">
      📅 2024-04-05
    </div>
    <details class="paper-abstract">
      Creating multilingual LLMs poses a significant challenge. Pretraining or fine-tuning LLMs to adopt new languages is evidently very costly. Furthermore, there exist limitations concerning benchmark datasets and the metrics used to measure model performance in multilingual settings. This paper proposes cost-effective solutions to both aforementioned challenges. Firstly, we introduce the Multilingual Instruction-Tuning Dataset (MITS), comprised of Alpaca-52K, Dolly-15K, and Vicuna Benchmark translations into 132 languages. Secondly, we propose a new method called \emph{TaCo: Translation-Assisted Cross-Linguality}, which utilizes translations in a chain-of-thought process to instruction-tune LLMs on new languages through a curriculum-learning process. As a proof of concept, we experimented with the instruction-tuned Guanaco-33B model, performing further instruction tuning using our proposed TaCo method in three low-resource languages and one high-resource language. Our results indicate that the TaCo method impresses GPT-4 with an 82\% score for a low-resource language in the Vicuna Benchmark dataset, doubling the performance in contrast to instruction tuning alone. Furthermore, TaCo shows promise in creating multilingual LLMs, even for low-resource languages. We have released our datasets and model adapters\footnote{https://github.com/UNHSAILLab/TaCo} , encouraging the research community to utilize these resources to advance work on multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13917v2">Could We Have Had Better Multilingual LLMs If English Was Not the Central Language?</a></div>
    <div class="paper-meta">
      📅 2024-04-05
      | 💬 TDLE 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong machine translation capabilities on languages they are trained on. However, the impact of factors beyond training data size on translation performance remains a topic of debate, especially concerning languages not directly encountered during training. Our study delves into Llama2's translation capabilities. By modeling a linear relationship between linguistic feature distances and machine translation scores, we ask ourselves if there are potentially better central languages for LLMs other than English. Our experiments show that the 7B Llama2 model yields above 10 BLEU when translating into all languages it has seen, which rarely happens for languages it has not seen. Most translation improvements into unseen languages come from scaling up the model size rather than instruction tuning or increasing shot count. Furthermore, our correlation analysis reveals that syntactic similarity is not the only linguistic factor that strongly correlates with machine translation scores. Interestingly, we discovered that under specific circumstances, some languages (e.g. Swedish, Catalan), despite having significantly less training data, exhibit comparable correlation levels to English. These insights challenge the prevailing landscape of LLMs, suggesting that models centered around languages other than English could provide a more efficient foundation for multilingual applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03891v1">Can only LLMs do Reasoning?: Potential of Small Language Models in Task Planning</a></div>
    <div class="paper-meta">
      📅 2024-04-05
      | 💬 8 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In robotics, the use of Large Language Models (LLMs) is becoming prevalent, especially for understanding human commands. In particular, LLMs are utilized as domain-agnostic task planners for high-level human commands. LLMs are capable of Chain-of-Thought (CoT) reasoning, and this allows LLMs to be task planners. However, we need to consider that modern robots still struggle to perform complex actions, and the domains where robots can be deployed are limited in practice. This leads us to pose a question: If small LMs can be trained to reason in chains within a single domain, would even small LMs be good task planners for the robots? To train smaller LMs to reason in chains, we build `COmmand-STeps datasets' (COST) consisting of high-level commands along with corresponding actionable low-level steps, via LLMs. We release not only our datasets but also the prompt templates used to generate them, to allow anyone to build datasets for their domain. We compare GPT3.5 and GPT4 with the finetuned GPT2 for task domains, in tabletop and kitchen environments, and the result shows that GPT2-medium is comparable to GPT3.5 for task planning in a specific domain. Our dataset, code, and more output samples can be found in https://github.com/Gawon-Choi/small-LMs-Task-Planning
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.10783v2">Zero- and Few-Shot Prompting with LLMs: A Comparative Study with Fine-tuned Models for Bangla Sentiment Analysis</a></div>
    <div class="paper-meta">
      📅 2024-04-05
      | 💬 Accepted at LREC-COLING 2024. Zero-Shot Prompting, Few-Shot Prompting, LLMs, Comparative Study, Fine-tuned Models, Bangla, Sentiment Analysis
    </div>
    <details class="paper-abstract">
      The rapid expansion of the digital world has propelled sentiment analysis into a critical tool across diverse sectors such as marketing, politics, customer service, and healthcare. While there have been significant advancements in sentiment analysis for widely spoken languages, low-resource languages, such as Bangla, remain largely under-researched due to resource constraints. Furthermore, the recent unprecedented performance of Large Language Models (LLMs) in various applications highlights the need to evaluate them in the context of low-resource languages. In this study, we present a sizeable manually annotated dataset encompassing 33,606 Bangla news tweets and Facebook comments. We also investigate zero- and few-shot in-context learning with several language models, including Flan-T5, GPT-4, and Bloomz, offering a comparative analysis against fine-tuned models. Our findings suggest that monolingual transformer-based models consistently outperform other models, even in zero and few-shot scenarios. To foster continued exploration, we intend to make this dataset and our research tools publicly available to the broader research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04302v1">CBR-RAG: Case-Based Reasoning for Retrieval Augmented Generation in LLMs for Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-04-04
      | 💬 Submitted to ICCBR'24
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enhances Large Language Model (LLM) output by providing prior knowledge as context to input. This is beneficial for knowledge-intensive and expert reliant tasks, including legal question-answering, which require evidence to validate generated text outputs. We highlight that Case-Based Reasoning (CBR) presents key opportunities to structure retrieval as part of the RAG process in an LLM. We introduce CBR-RAG, where CBR cycle's initial retrieval stage, its indexing vocabulary, and similarity knowledge containers are used to enhance LLM queries with contextually relevant cases. This integration augments the original LLM query, providing a richer prompt. We present an evaluation of CBR-RAG, and examine different representations (i.e. general and domain-specific embeddings) and methods of comparison (i.e. inter, intra and hybrid similarity) on the task of legal question-answering. Our results indicate that the context provided by CBR's case reuse enforces similarity between relevant components of the questions and the evidence base leading to significant improvements in the quality of generated answers.
    </details>
</div>
