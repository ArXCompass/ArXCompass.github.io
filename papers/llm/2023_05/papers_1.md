# llm - 2023_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.00029v1">CodeTF: One-stop Transformer Library for State-of-the-art Code LLM</a></div>
    <div class="paper-meta">
      📅 2023-05-31
      | 💬 Ongoing work - Draft Preview
    </div>
    <details class="paper-abstract">
      Code intelligence plays a key role in transforming modern software engineering. Recently, deep learning-based models, especially Transformer-based large language models (LLMs), have demonstrated remarkable potential in tackling these tasks by leveraging massive open-source code data and programming language features. However, the development and deployment of such models often require expertise in both machine learning and software engineering, creating a barrier for the model adoption. In this paper, we present CodeTF, an open-source Transformer-based library for state-of-the-art Code LLMs and code intelligence. Following the principles of modular design and extensible framework, we design CodeTF with a unified interface to enable rapid access and development across different types of models, datasets and tasks. Our library supports a collection of pretrained Code LLM models and popular code benchmarks, including a standardized interface to train and serve code LLMs efficiently, and data features such as language-specific parsers and utility functions for extracting code attributes. In this paper, we describe the design principles, the architecture, key modules and components, and compare with other related library tools. Finally, we hope CodeTF is able to bridge the gap between machine learning/generative AI and software engineering, providing a comprehensive open-source solution for developers, researchers, and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2305.13144v2">Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline</a></div>
    <div class="paper-meta">
      📅 2023-05-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.13731v2">Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model</a></div>
    <div class="paper-meta">
      📅 2023-05-29
      | 💬 https://github.com/declare-lab/tango
    </div>
    <details class="paper-abstract">
      The immense scale of the recent large language models (LLM) allows many interesting properties, such as, instruction- and chain-of-thought-based fine-tuning, that has significantly improved zero- and few-shot performance in many natural language processing (NLP) tasks. Inspired by such successes, we adopt such an instruction-tuned LLM Flan-T5 as the text encoder for text-to-audio (TTA) generation -- a task where the goal is to generate an audio from its textual description. The prior works on TTA either pre-trained a joint text-audio encoder or used a non-instruction-tuned model, such as, T5. Consequently, our latent diffusion model (LDM)-based approach TANGO outperforms the state-of-the-art AudioLDM on most metrics and stays comparable on the rest on AudioCaps test set, despite training the LDM on a 63 times smaller dataset and keeping the text encoder frozen. This improvement might also be attributed to the adoption of audio pressure level-based sound mixing for training set augmentation, whereas the prior methods take a random mix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13144v2">Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline</a></div>
    <div class="paper-meta">
      📅 2023-05-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16633v1">Zero is Not Hero Yet: Benchmarking Zero-Shot Performance of LLMs for Financial Tasks</a></div>
    <div class="paper-meta">
      📅 2023-05-26
      | 💬 Working Paper
    </div>
    <details class="paper-abstract">
      Recently large language models (LLMs) like ChatGPT have shown impressive performance on many natural language processing tasks with zero-shot. In this paper, we investigate the effectiveness of zero-shot LLMs in the financial domain. We compare the performance of ChatGPT along with some open-source generative LLMs in zero-shot mode with RoBERTa fine-tuned on annotated data. We address three inter-related research questions on data annotation, performance gaps, and the feasibility of employing generative models in the finance domain. Our findings demonstrate that ChatGPT performs well even without labeled data but fine-tuned models generally outperform it. Our research also highlights how annotating with generative models can be time-intensive. Our codebase is publicly available on GitHub under CC BY-NC 4.0 license.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.08302v3">ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation</a></div>
    <div class="paper-meta">
      📅 2023-05-26
      | 💬 25 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a promising technique for mitigating memory consumption and computational costs in large language models (LLMs). However, a systematic examination of various quantization schemes, model families, and quantization bit precision has been absent from the literature. In this paper, we conduct a comprehensive analysis of these factors by investigating the effects of PTQ on weight-only, activation-only, and weight-and-activation quantization using diverse methods such as round-to-nearest (RTN), GPTQ, ZeroQuant, and their variants. We apply these methods to two distinct model families with parameters ranging from 125M to 176B. Our contributions include: (1) a sensitivity analysis revealing that activation quantization is generally more susceptible to weight quantization, with smaller models often outperforming larger models in terms of activation quantization; (2) an evaluation and comparison of existing PTQ methods to optimize model size reduction while minimizing the impact on accuracy, revealing that none of the current methods can achieve the original model quality for quantization with either INT4-weight or INT4-weight-and-INT8-activation; (3) based on these insights, we propose an optimized method called Low-Rank Compensation (LoRC), which employs low-rank matrices to enhance model quality recovery with a minimal increase in model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.15808v1">Towards Language-guided Interactive 3D Generation: LLMs as Layout Interpreter with Generative Feedback</a></div>
    <div class="paper-meta">
      📅 2023-05-25
      | 💬 Preprint. Work in Progres
    </div>
    <details class="paper-abstract">
      Generating and editing a 3D scene guided by natural language poses a challenge, primarily due to the complexity of specifying the positional relations and volumetric changes within the 3D space. Recent advancements in Large Language Models (LLMs) have demonstrated impressive reasoning, conversational, and zero-shot generation abilities across various domains. Surprisingly, these models also show great potential in realizing and interpreting the 3D space. In light of this, we propose a novel language-guided interactive 3D generation system, dubbed LI3D, that integrates LLMs as a 3D layout interpreter into the off-the-shelf layout-to-3D generative models, allowing users to flexibly and interactively generate visual content. Specifically, we design a versatile layout structure base on the bounding boxes and semantics to prompt the LLMs to model the spatial generation and reasoning from language. Our system also incorporates LLaVA, a large language and vision assistant, to provide generative feedback from the visual aspect for improving the visual quality of generated content. We validate the effectiveness of LI3D, primarily in 3D generation and editing through multi-round interactions, which can be flexibly extended to 2D generation and editing. Various experiments demonstrate the potential benefits of incorporating LLMs in generative AI for applications, e.g., metaverse. Moreover, we benchmark the layout reasoning performance of LLMs with neural visual artist tasks, revealing their emergent ability in the spatial layout domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14693v1">Have Large Language Models Developed a Personality?: Applicability of Self-Assessment Tests in Measuring Personality in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-05-24
    </div>
    <details class="paper-abstract">
      Have Large Language Models (LLMs) developed a personality? The short answer is a resounding "We Don't Know!". In this paper, we show that we do not yet have the right tools to measure personality in language models. Personality is an important characteristic that influences behavior. As LLMs emulate human-like intelligence and performance in various tasks, a natural question to ask is whether these models have developed a personality. Previous works have evaluated machine personality through self-assessment personality tests, which are a set of multiple-choice questions created to evaluate personality in humans. A fundamental assumption here is that human personality tests can accurately measure personality in machines. In this paper, we investigate the emergence of personality in five LLMs of different sizes ranging from 1.5B to 30B. We propose the Option-Order Symmetry property as a necessary condition for the reliability of these self-assessment tests. Under this condition, the answer to self-assessment questions is invariant to the order in which the options are presented. We find that many LLMs personality test responses do not preserve option-order symmetry. We take a deeper look at LLMs test responses where option-order symmetry is preserved to find that in these cases, LLMs do not take into account the situational statement being tested and produce the exact same answer irrespective of the situation being tested. We also identify the existence of inherent biases in these LLMs which is the root cause of the aforementioned phenomenon and makes self-assessment tests unreliable. These observations indicate that self-assessment tests are not the correct tools to measure personality in LLMs. Through this paper, we hope to draw attention to the shortcomings of current literature in measuring personality in LLMs and call for developing tools for machine personality measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14540v1">LLMs as Factual Reasoners: Insights from Existing Benchmarks and Beyond</a></div>
    <div class="paper-meta">
      📅 2023-05-23
    </div>
    <details class="paper-abstract">
      With the recent appearance of LLMs in practical settings, having methods that can effectively detect factual inconsistencies is crucial to reduce the propagation of misinformation and improve trust in model outputs. When testing on existing factual consistency benchmarks, we find that a few large language models (LLMs) perform competitively on classification benchmarks for factual inconsistency detection compared to traditional non-LLM methods. However, a closer analysis reveals that most LLMs fail on more complex formulations of the task and exposes issues with existing evaluation benchmarks, affecting evaluation precision. To address this, we propose a new protocol for inconsistency detection benchmark creation and implement it in a 10-domain benchmark called SummEdits. This new benchmark is 20 times more cost-effective per sample than previous benchmarks and highly reproducible, as we estimate inter-annotator agreement at about 0.9. Most LLMs struggle on SummEdits, with performance close to random chance. The best-performing model, GPT-4, is still 8\% below estimated human performance, highlighting the gaps in LLMs' ability to reason about facts and detect inconsistencies when they occur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.16334v1">OlaGPT: Empowering LLMs With Human-like Problem-Solving Abilities</a></div>
    <div class="paper-meta">
      📅 2023-05-23
    </div>
    <details class="paper-abstract">
      In most current research, large language models (LLMs) are able to perform reasoning tasks by generating chains of thought through the guidance of specific prompts. However, there still exists a significant discrepancy between their capability in solving complex reasoning problems and that of humans. At present, most approaches focus on chains of thought (COT) and tool use, without considering the adoption and application of human cognitive frameworks. It is well-known that when confronting complex reasoning challenges, humans typically employ various cognitive abilities, and necessitate interaction with all aspects of tools, knowledge, and the external environment information to accomplish intricate tasks. This paper introduces a novel intelligent framework, referred to as OlaGPT. OlaGPT carefully studied a cognitive architecture framework, and propose to simulate certain aspects of human cognition. The framework involves approximating different cognitive modules, including attention, memory, reasoning, learning, and corresponding scheduling and decision-making mechanisms. Inspired by the active learning mechanism of human beings, it proposes a learning unit to record previous mistakes and expert opinions, and dynamically refer to them to strengthen their ability to solve similar problems. The paper also outlines common effective reasoning frameworks for human problem-solving and designs Chain-of-Thought (COT) templates accordingly. A comprehensive decision-making mechanism is also proposed to maximize model accuracy. The efficacy of OlaGPT has been stringently evaluated on multiple reasoning datasets, and the experimental outcomes reveal that OlaGPT surpasses state-of-the-art benchmarks, demonstrating its superior performance. Our implementation of OlaGPT is available on GitHub: \url{https://github.com/oladata-team/OlaGPT}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13102v1">Observations on LLMs for Telecom Domain: Capabilities and Limitations</a></div>
    <div class="paper-meta">
      📅 2023-05-22
      | 💬 11 pages, 2 figures, 8 tables
    </div>
    <details class="paper-abstract">
      The landscape for building conversational interfaces (chatbots) has witnessed a paradigm shift with recent developments in generative Artificial Intelligence (AI) based Large Language Models (LLMs), such as ChatGPT by OpenAI (GPT3.5 and GPT4), Google's Bard, Large Language Model Meta AI (LLaMA), among others. In this paper, we analyze capabilities and limitations of incorporating such models in conversational interfaces for the telecommunication domain, specifically for enterprise wireless products and services. Using Cradlepoint's publicly available data for our experiments, we present a comparative analysis of the responses from such models for multiple use-cases including domain adaptation for terminology and product taxonomy, context continuity, robustness to input perturbations and errors. We believe this evaluation would provide useful insights to data scientists engaged in building customized conversational interfaces for domain-specific requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12799v1">Interactive Data Synthesis for Systematic Vision Adaptation via LLMs-AIGCs Collaboration</a></div>
    <div class="paper-meta">
      📅 2023-05-22
      | 💬 11 pages, 6 figures, technical report
    </div>
    <details class="paper-abstract">
      Recent text-to-image generation models have shown promising results in generating high-fidelity photo-realistic images. In parallel, the problem of data scarcity has brought a growing interest in employing AIGC technology for high-quality data expansion. However, this paradigm requires well-designed prompt engineering that cost-less data expansion and labeling remain under-explored. Inspired by LLM's powerful capability in task guidance, we propose a new paradigm of annotated data expansion named as ChatGenImage. The core idea behind it is to leverage the complementary strengths of diverse models to establish a highly effective and user-friendly pipeline for interactive data augmentation. In this work, we extensively study how LLMs communicate with AIGC model to achieve more controllable image generation and make the first attempt to collaborate them for automatic data augmentation for a variety of downstream tasks. Finally, we present fascinating results obtained from our ChatGenImage framework and demonstrate the powerful potential of our synthetic data for systematic vision adaptation. Our codes are available at https://github.com/Yuqifan1117/Labal-Anything-Pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12720v1">llm-japanese-dataset v0: Construction of Japanese Chat Dataset for Large Language Models and its Methodology</a></div>
    <div class="paper-meta">
      📅 2023-05-22
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      This study constructed a Japanese chat dataset for tuning large language models (LLMs), which consist of about 8.4 million records. Recently, LLMs have been developed and gaining popularity. However, high-performing LLMs are usually mainly for English. There are two ways to support languages other than English by those LLMs: constructing LLMs from scratch or tuning existing models. However, in both ways, datasets are necessary parts. In this study, we focused on supporting Japanese in those LLMs and making a dataset for training or tuning LLMs in Japanese. The dataset we constructed consisted of various tasks, such as translation and knowledge tasks. In our experiment, we tuned an existing LLM using our dataset and evaluated the performance qualitatively. The results suggest that our dataset is possibly beneficial for LLMs. However, we also revealed some difficulties in constructing LLMs in languages other than English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11334v1">Writing your own book: A method for going from closed to open book QA to improve robustness and performance of smaller LLMs</a></div>
    <div class="paper-meta">
      📅 2023-05-18
    </div>
    <details class="paper-abstract">
      We introduce two novel methods, Tree-Search and Self-contextualizing QA, designed to enhance the performance of large language models (LLMs) in question-answering tasks. Tree-Search is a sampling technique specifically created to extract diverse information from an LLM for a given prompt. Self-contextualizing QA leverages Tree-Search to enable the model to create its own context using a wide range of information relevant to the prompt, evaluate it explicitly and return a open book answer to the initial prompt . We demonstrate that the quality of generated answers improves according to various metrics, including accuracy, informativeness, coherence, and consistency, as evaluated by GPT3.5(text-davinci-003). Furthermore, we show that our methods result in increased robustness and that performance is positively correlated with tree size, benefiting both answer quality and robustness. Finally, we discuss other promising applications of Tree-Search, highlighting its potential to enhance a broad range of tasks beyond question-answering. \noindent We also discuss several areas for future work, including refining the Tree-Search and Self-Contextualizing QA methods, improving the coherence of the generated context, and investigating the impact of bootstrapping on model robustness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.09858v1">Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-05-17
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) play a crucial role in enhancing e-commerce system performance by providing structured information about entities and their relationships, such as complementary or substitutable relations between products or product types, which can be utilized in recommender systems. However, relation labeling in KGs remains a challenging task due to the dynamic nature of e-commerce domains and the associated cost of human labor. Recently, breakthroughs in Large Language Models (LLMs) have shown surprising results in numerous natural language processing tasks. In this paper, we conduct an empirical study of LLMs for relation labeling in e-commerce KGs, investigating their powerful learning capabilities in natural language and effectiveness in predicting relations between product types with limited labeled data. We evaluate various LLMs, including PaLM and GPT-3.5, on benchmark datasets, demonstrating their ability to achieve competitive performance compared to humans on relation labeling tasks using just 1 to 5 labeled examples per relation. Additionally, we experiment with different prompt engineering techniques to examine their impact on model performance. Our results show that LLMs significantly outperform existing KG completion models in relation labeling for e-commerce KGs and exhibit performance strong enough to replace human labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.09067v1">SGP-TOD: Building Task Bots Effortlessly via Schema-Guided LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2023-05-15
      | 💬 21 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Building end-to-end task bots and maintaining their integration with new functionalities using minimal human efforts is a long-standing challenge in dialog research. Recently large language models (LLMs) have demonstrated exceptional proficiency in conversational engagement and adherence to instructions across various downstream tasks. In this work, we introduce SGP-TOD, Schema-Guided Prompting for building Task-Oriented Dialog systems effortlessly based on LLMs. Utilizing the symbolic knowledge -- task schema, we instruct fixed LLMs to generate appropriate responses on novel tasks, circumventing the need for training data. Specifically, SGP-TOD comprises three components: a LLM for engaging with users, a DST Prompter to aid the LLM with dialog state tracking, which is then used to retrieve database items, and a Policy Prompter to elicit proper responses adhering to the provided dialog policy. Experimental results on Multiwoz, RADDLE and STAR datasets show that our training-free strategy SGP-TOD, without any task-specific data, yields state-of-the-art (SOTA) zero-shot performance, greatly surpasses the few-shot approaches. In a domain-extension setting, SGP-TOD aptly adapts to new functionalities by merely adding supplementary schema rules. We make our code and data publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05182v1">Interactive Fashion Content Generation Using LLMs and Latent Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2023-05-15
      | 💬 Third Workshop on Ethical Considerations in Creative applications of Computer Vision (EC3V) at CVPR 2023. arXiv admin note: substantial text overlap with arXiv:2301.02110, arXiv:2112.10752 by other authors
    </div>
    <details class="paper-abstract">
      Fashionable image generation aims to synthesize images of diverse fashion prevalent around the globe, helping fashion designers in real-time visualization by giving them a basic customized structure of how a specific design preference would look in real life and what further improvements can be made for enhanced customer satisfaction. Moreover, users can alone interact and generate fashionable images by just giving a few simple prompts. Recently, diffusion models have gained popularity as generative models owing to their flexibility and generation of realistic images from Gaussian noise. Latent diffusion models are a type of generative model that use diffusion processes to model the generation of complex data, such as images, audio, or text. They are called "latent" because they learn a hidden representation, or latent variable, of the data that captures its underlying structure. We propose a method exploiting the equivalence between diffusion models and energy-based models (EBMs) and suggesting ways to compose multiple probability distributions. We describe a pipeline on how our method can be used specifically for new fashionable outfit generation and virtual try-on using LLM-guided text-to-image generation. Our results indicate that using an LLM to refine the prompts to the latent diffusion model assists in generating globally creative and culturally diversified fashion styles and reducing bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2305.06474v1">Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction</a></div>
    <div class="paper-meta">
      📅 2023-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities in generalizing to new tasks in a zero-shot or few-shot manner. However, the extent to which LLMs can comprehend user preferences based on their previous behavior remains an emerging and still unclear research question. Traditionally, Collaborative Filtering (CF) has been the most effective method for these tasks, predominantly relying on the extensive volume of rating data. In contrast, LLMs typically demand considerably less data while maintaining an exhaustive world knowledge about each item, such as movies or products. In this paper, we conduct a thorough examination of both CF and LLMs within the classic task of user rating prediction, which involves predicting a user's rating for a candidate item based on their past ratings. We investigate various LLMs in different sizes, ranging from 250M to 540B parameters and evaluate their performance in zero-shot, few-shot, and fine-tuning scenarios. We conduct comprehensive analysis to compare between LLMs and strong CF methods, and find that zero-shot LLMs lag behind traditional recommender models that have the access to user interaction data, indicating the importance of user interaction data. However, through fine-tuning, LLMs achieve comparable or even better performance with only a small fraction of the training data, demonstrating their potential through data efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.11116v3">Graph-ToolFormer: To Empower LLMs with Graph Reasoning Ability via Prompt Augmented by ChatGPT</a></div>
    <div class="paper-meta">
      📅 2023-05-11
      | 💬 34 pages, 3 figures, 8 tables
    </div>
    <details class="paper-abstract">
      In this paper, we aim to develop a large language model (LLM) with the reasoning ability on complex graph data. Currently, LLMs have achieved very impressive performance on various natural language learning tasks, extensions of which have also been applied to study the vision tasks with multi-modal data. However, when it comes to the graph learning tasks, existing LLMs present very serious flaws due to their several inherited weaknesses in performing {multi-step logic reasoning}, {precise mathematical calculation} and {perception about the spatial and temporal factors}. To address such challenges, in this paper, we will investigate the principles, methodologies and algorithms to empower existing LLMs with graph reasoning ability, which will have tremendous impacts on the current research of both LLMs and graph learning. Inspired by the latest ChatGPT and Toolformer models, we propose the Graph-ToolFormer (Graph Reasoning oriented Toolformer) framework to teach LLMs themselves with prompts augmented by ChatGPT to use external graph reasoning API tools. Specifically, we will investigate to teach Graph-ToolFormer to handle various graph data reasoning tasks in this paper, including both (1) very basic graph data loading and graph property reasoning tasks, ranging from simple graph order and size to the graph diameter and periphery, and (2) more advanced reasoning tasks on real-world graph data, such as bibliographic networks, protein molecules, sequential recommender systems, social networks and knowledge graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06474v1">Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction</a></div>
    <div class="paper-meta">
      📅 2023-05-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities in generalizing to new tasks in a zero-shot or few-shot manner. However, the extent to which LLMs can comprehend user preferences based on their previous behavior remains an emerging and still unclear research question. Traditionally, Collaborative Filtering (CF) has been the most effective method for these tasks, predominantly relying on the extensive volume of rating data. In contrast, LLMs typically demand considerably less data while maintaining an exhaustive world knowledge about each item, such as movies or products. In this paper, we conduct a thorough examination of both CF and LLMs within the classic task of user rating prediction, which involves predicting a user's rating for a candidate item based on their past ratings. We investigate various LLMs in different sizes, ranging from 250M to 540B parameters and evaluate their performance in zero-shot, few-shot, and fine-tuning scenarios. We conduct comprehensive analysis to compare between LLMs and strong CF methods, and find that zero-shot LLMs lag behind traditional recommender models that have the access to user interaction data, indicating the importance of user interaction data. However, through fine-tuning, LLMs achieve comparable or even better performance with only a small fraction of the training data, demonstrating their potential through data efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2208.11981v2">On Reality and the Limits of Language Data: Aligning LLMs with Human Norms</a></div>
    <div class="paper-meta">
      📅 2023-05-09
      | 💬 9 pages; data available, see https://sites.google.com/site/nhcollier/projects/art
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) harness linguistic associations in vast natural language data for practical applications. However, their ability to understand the physical world using only language data remains a question. After reviewing existing protocols, we explore this question using a novel and tightly controlled reasoning test (ART) and compare human norms against versions of GPT-3. Our findings highlight the categories of common-sense relations models that could learn directly from data and areas of weakness. GPT-3 offers evidence for verbal reasoning on a par with human subjects for several relations including Synonymy, Antonymy, and Default inheritance, Without reinforcement learning from human judgements, it appears GPT-3 performs at the lower end of the reference interval for Has-part and Contained-in. Weaknesses were observed also in affordance characteristics through Necessary-quality, Order-of-size and Order-of-intensity. Combining LLMs with symbolic world grounding is a promising direction to address associative learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.05138v1">Read, Diagnose and Chat: Towards Explainable and Interactive LLMs-Augmented Depression Detection in Social Media</a></div>
    <div class="paper-meta">
      📅 2023-05-09
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This paper proposes a new depression detection system based on LLMs that is both interpretable and interactive. It not only provides a diagnosis, but also diagnostic evidence and personalized recommendations based on natural language dialogue with the user. We address challenges such as the processing of large amounts of text and integrate professional diagnostic criteria. Our system outperforms traditional methods across various settings and is demonstrated through case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.04533v1">Prompted LLMs as Chatbot Modules for Long Open-domain Conversation</a></div>
    <div class="paper-meta">
      📅 2023-05-08
      | 💬 Accepted to the Findings of ACL2023. The camera-ready version with additional experimental results will be uploaded
    </div>
    <details class="paper-abstract">
      In this paper, we propose MPC (Modular Prompted Chatbot), a new approach for creating high-quality conversational agents without the need for fine-tuning. Our method utilizes pre-trained large language models (LLMs) as individual modules for long-term consistency and flexibility, by using techniques such as few-shot prompting, chain-of-thought (CoT), and external memory. Our human evaluation results show that MPC is on par with fine-tuned chatbot models in open-domain conversations, making it an effective solution for creating consistent and engaging chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2302.12173v2">Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2023-05-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various applications. The functionalities of recent LLMs can be flexibly modulated via natural language prompts. This renders them susceptible to targeted adversarial prompting, e.g., Prompt Injection (PI) attacks enable attackers to override original instructions and employed controls. So far, it was assumed that the user is directly prompting the LLM. But, what if it is not the user prompting? We argue that LLM-Integrated Applications blur the line between data and instructions. We reveal new attack vectors, using Indirect Prompt Injection, that enable adversaries to remotely (without a direct interface) exploit LLM-integrated applications by strategically injecting prompts into data likely to be retrieved. We derive a comprehensive taxonomy from a computer security perspective to systematically investigate impacts and vulnerabilities, including data theft, worming, information ecosystem contamination, and other novel security risks. We demonstrate our attacks' practical viability against both real-world systems, such as Bing's GPT-4 powered Chat and code-completion engines, and synthetic applications built on GPT-4. We show how processing retrieved prompts can act as arbitrary code execution, manipulate the application's functionality, and control how and if other APIs are called. Despite the increasing integration and reliance on LLMs, effective mitigations of these emerging threats are currently lacking. By raising awareness of these vulnerabilities and providing key insights into their implications, we aim to promote the safe and responsible deployment of these powerful models and the development of robust defenses that protect users and systems from potential attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.03300v1">LLM-RM at SemEval-2023 Task 2: Multilingual Complex NER using XLM-RoBERTa</a></div>
    <div class="paper-meta">
      📅 2023-05-05
      | 💬 Submitted to SemEval-2023, The 17th International Workshop on Semantic Evaluation
    </div>
    <details class="paper-abstract">
      Named Entity Recognition(NER) is a task of recognizing entities at a token level in a sentence. This paper focuses on solving NER tasks in a multilingual setting for complex named entities. Our team, LLM-RM participated in the recently organized SemEval 2023 task, Task 2: MultiCoNER II,Multilingual Complex Named Entity Recognition. We approach the problem by leveraging cross-lingual representation provided by fine-tuning XLM-Roberta base model on datasets of all of the 12 languages provided -- Bangla, Chinese, English, Farsi, French, German, Hindi, Italian, Portuguese, Spanish, Swedish and Ukrainian
    </details>
</div>
