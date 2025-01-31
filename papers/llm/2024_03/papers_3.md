# llm - 2024_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18571v3">Arithmetic Control of LLMs for Diverse User Preferences: Directional Preference Alignment with Multi-Objective Rewards</a></div>
    <div class="paper-meta">
      📅 2024-03-06
      | 💬 The code and model are released at https://github.com/Haoxiang-Wang/directional-preference-alignment
    </div>
    <details class="paper-abstract">
      Fine-grained control over large language models (LLMs) remains a significant challenge, hindering their adaptability to diverse user needs. While Reinforcement Learning from Human Feedback (RLHF) shows promise in aligning LLMs, its reliance on scalar rewards often limits its ability to capture diverse user preferences in real-world applications. To address this limitation, we introduce the Directional Preference Alignment (DPA) framework. Unlike the scalar-reward RLHF, DPA incorporates multi-objective reward modeling to represent diverse preference profiles. Additionally, DPA models user preferences as directions (i.e., unit vectors) in the reward space to achieve user-dependent preference control. Our method involves training a multi-objective reward model and then fine-tuning the LLM with a preference-conditioned variant of Rejection Sampling Finetuning (RSF), an RLHF method adopted by Llama 2. This method enjoys a better performance trade-off across various reward objectives. In comparison with the scalar-reward RLHF, DPA offers users intuitive control over LLM generation: they can arithmetically specify their desired trade-offs (e.g., more helpfulness with less verbosity). We also validate the effectiveness of DPA with real-world alignment experiments on Mistral-7B. Our method provides straightforward arithmetic control over the trade-off between helpfulness and verbosity while maintaining competitive performance with strong baselines such as Direct Preference Optimization (DPO).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.12357v2">Large Language Models for Code Analysis: Do LLMs Really Do Their Job?</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 Accepted by Usenix Security 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in the realm of natural language understanding and programming code processing tasks. Their capacity to comprehend and generate human-like code has spurred research into harnessing LLMs for code analysis purposes. However, the existing body of literature falls short in delivering a systematic evaluation and assessment of LLMs' effectiveness in code analysis, particularly in the context of obfuscated code. This paper seeks to bridge this gap by offering a comprehensive evaluation of LLMs' capabilities in performing code analysis tasks. Additionally, it presents real-world case studies that employ LLMs for code analysis. Our findings indicate that LLMs can indeed serve as valuable tools for automating code analysis, albeit with certain limitations. Through meticulous exploration, this research contributes to a deeper understanding of the potential and constraints associated with utilizing LLMs in code analysis, paving the way for enhanced applications in this critical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03288v1">Should We Fear Large Language Models? A Structural Analysis of the Human Reasoning System for Elucidating LLM Capabilities and Risks Through the Lens of Heidegger's Philosophy</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 39 pages
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of Large Language Models (LLMs), there is a critical need to thoroughly analyze their capabilities and risks. Central to our investigation are two novel elements. Firstly, it is the innovative parallels between the statistical patterns of word relationships within LLMs and Martin Heidegger's concepts of "ready-to-hand" and "present-at-hand," which encapsulate the utilitarian and scientific altitudes humans employ in interacting with the world. This comparison lays the groundwork for positioning LLMs as the digital counterpart to the Faculty of Verbal Knowledge, shedding light on their capacity to emulate certain facets of human reasoning. Secondly, a structural analysis of human reasoning, viewed through Heidegger's notion of truth as "unconcealment" is conducted This foundational principle enables us to map out the inputs and outputs of the reasoning system and divide reasoning into four distinct categories. Respective cognitive faculties are delineated, allowing us to place LLMs within the broader schema of human reasoning, thus clarifying their strengths and inherent limitations. Our findings reveal that while LLMs possess the capability for Direct Explicative Reasoning and Pseudo Rational Reasoning, they fall short in authentic rational reasoning and have no creative reasoning capabilities, due to the current lack of many analogous AI models such as the Faculty of Judgement. The potential and risks of LLMs when they are augmented with other AI technologies are also evaluated. The results indicate that although LLMs have achieved proficiency in some reasoning abilities, the aspiration to match or exceed human intellectual capabilities is yet unattained. This research not only enriches our comprehension of LLMs but also propels forward the discourse on AI's potential and its bounds, paving the way for future explorations into AI's evolving landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03017v1">OPEx: A Component-Wise Analysis of LLM-Centric Agents in Embodied Instruction Following</a></div>
    <div class="paper-meta">
      📅 2024-03-05
    </div>
    <details class="paper-abstract">
      Embodied Instruction Following (EIF) is a crucial task in embodied learning, requiring agents to interact with their environment through egocentric observations to fulfill natural language instructions. Recent advancements have seen a surge in employing large language models (LLMs) within a framework-centric approach to enhance performance in embodied learning tasks, including EIF. Despite these efforts, there exists a lack of a unified understanding regarding the impact of various components-ranging from visual perception to action execution-on task performance. To address this gap, we introduce OPEx, a comprehensive framework that delineates the core components essential for solving embodied learning tasks: Observer, Planner, and Executor. Through extensive evaluations, we provide a deep analysis of how each component influences EIF task performance. Furthermore, we innovate within this space by deploying a multi-agent dialogue strategy on a TextWorld counterpart, further enhancing task performance. Our findings reveal that LLM-centric design markedly improves EIF outcomes, identify visual perception and low-level action execution as critical bottlenecks, and demonstrate that augmenting LLMs with a multi-agent framework further elevates performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03008v1">Knowledge Graphs as Context Sources for LLM-Based Explanations of Learning Recommendations</a></div>
    <div class="paper-meta">
      📅 2024-03-05
    </div>
    <details class="paper-abstract">
      In the era of personalized education, the provision of comprehensible explanations for learning recommendations is of a great value to enhance the learner's understanding and engagement with the recommended learning content. Large language models (LLMs) and generative AI in general have recently opened new doors for generating human-like explanations, for and along learning recommendations. However, their precision is still far away from acceptable in a sensitive field like education. To harness the abilities of LLMs, while still ensuring a high level of precision towards the intent of the learners, this paper proposes an approach to utilize knowledge graphs (KG) as a source of factual context, for LLM prompts, reducing the risk of model hallucinations, and safeguarding against wrong or imprecise information, while maintaining an application-intended learning context. We utilize the semantic relations in the knowledge graph to offer curated knowledge about learning recommendations. With domain-experts in the loop, we design the explanation as a textual template, which is filled and completed by the LLM. Domain experts were integrated in the prompt engineering phase as part of a study, to ensure that explanations include information that is relevant to the learner. We evaluate our approach quantitatively using Rouge-N and Rouge-L measures, as well as qualitatively with experts and learners. Our results show an enhanced recall and precision of the generated explanations compared to those generated solely by the GPT model, with a greatly reduced risk of generating imprecise information in the final learning explanation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02901v1">A Comprehensive Survey on Process-Oriented Automatic Text Summarization with Exploration of LLM-Based Methods</a></div>
    <div class="paper-meta">
      📅 2024-03-05
    </div>
    <details class="paper-abstract">
      Automatic Text Summarization (ATS), utilizing Natural Language Processing (NLP) algorithms, aims to create concise and accurate summaries, thereby significantly reducing the human effort required in processing large volumes of text. ATS has drawn considerable interest in both academic and industrial circles. Many studies have been conducted in the past to survey ATS methods; however, they generally lack practicality for real-world implementations, as they often categorize previous methods from a theoretical standpoint. Moreover, the advent of Large Language Models (LLMs) has altered conventional ATS methods. In this survey, we aim to 1) provide a comprehensive overview of ATS from a ``Process-Oriented Schema'' perspective, which is best aligned with real-world implementations; 2) comprehensively review the latest LLM-based ATS works; and 3) deliver an up-to-date survey of ATS, bridging the two-year gap in the literature. To the best of our knowledge, this is the first survey to specifically investigate LLM-based ATS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06509v3">AntEval: Evaluation of Social Interaction Competencies in LLM-Driven Agents</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 Preliminary version of an ongoing work
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated their ability to replicate human behaviors across a wide range of scenarios. However, their capability in handling complex, multi-character social interactions has yet to be fully explored, primarily due to the absence of robust, quantitative evaluation methods. This gap has slowed the development of agents proficient in more nuanced interactions beyond simple exchanges, for example, small talk. To address this challenge, we introduce the Multi-Agent Interaction Evaluation Framework (AntEval), encompassing a novel interaction framework and evaluation methods. The interaction framework aims to foster an complex interaction environment that bolsters information exchange and intention expression within social interactions. Furthermore, we introduce evaluation methods, including two metrics: Information Exchanging Precision (IEP) and Interaction Expressiveness Gap (IEG), designed for the quantitative and objective assessment of agents' interaction competencies. Our findings highlight the utility of these evaluative methods and show significant potential for improving LLMs' ability to construct agents that interact in a more natural manner with human-like intricacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02775v1">EasyQuant: An Efficient Data-free Quantization Algorithm for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have proven to be very superior to conventional methods in various tasks. However, their expensive computations and high memory requirements are prohibitive for deployment. Model quantization is an effective method for reducing this overhead. The problem is that in most previous works, the quantized model was calibrated using few samples from the training data, which might affect the generalization of the quantized LLMs to unknown cases and tasks. Hence in this work, we explore an important question: Can we design a data-independent quantization method for LLMs to guarantee its generalization performance? In this work, we propose EasyQuant, a training-free and data-independent weight-only quantization algorithm for LLMs. Our observation indicates that two factors: outliers in the weight and quantization ranges, are essential for reducing the quantization error. Therefore, in EasyQuant, we leave the outliers (less than 1%) unchanged and optimize the quantization range to reduce the reconstruction error. With these methods, we surprisingly find that EasyQuant achieves comparable performance to the original model. Since EasyQuant does not depend on any training data, the generalization performance of quantized LLMs is safely guaranteed. Moreover, EasyQuant can be implemented in parallel so that the quantized model could be attained in a few minutes even for LLMs over 100B. To our best knowledge, we are the first work that achieves almost lossless quantization performance for LLMs under a data-independent setting and our algorithm runs over 10 times faster than the data-dependent methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02727v1">HARGPT: Are LLMs Zero-Shot Human Activity Recognizers?</a></div>
    <div class="paper-meta">
      📅 2024-03-05
    </div>
    <details class="paper-abstract">
      There is an ongoing debate regarding the potential of Large Language Models (LLMs) as foundational models seamlessly integrated with Cyber-Physical Systems (CPS) for interpreting the physical world. In this paper, we carry out a case study to answer the following question: Are LLMs capable of zero-shot human activity recognition (HAR). Our study, HARGPT, presents an affirmative answer by demonstrating that LLMs can comprehend raw IMU data and perform HAR tasks in a zero-shot manner, with only appropriate prompts. HARGPT inputs raw IMU data into LLMs and utilizes the role-play and think step-by-step strategies for prompting. We benchmark HARGPT on GPT4 using two public datasets of different inter-class similarities and compare various baselines both based on traditional machine learning and state-of-the-art deep classification models. Remarkably, LLMs successfully recognize human activities from raw IMU data and consistently outperform all the baselines on both datasets. Our findings indicate that by effective prompting, LLMs can interpret raw IMU data based on their knowledge base, possessing a promising potential to analyze raw sensor data of the physical world effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14160v2">Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 82 pages, 9 figures, 54 tables
    </div>
    <details class="paper-abstract">
      Speculative decoding is an inference-acceleration method for large language models (LLMs) where a small language model generates a draft-token sequence which is further verified by the target LLM in parallel. Recent works have advanced this method by establishing a draft-token tree, achieving superior performance over a single-sequence speculative decoding. However, those works independently generate tokens at each level of the tree, not leveraging the tree's entire diversifiability. Besides, their empirical superiority has been shown for fixed length of sequences, implicitly granting more computational resource to LLM for the tree-based methods. None of the existing works has conducted empirical studies with fixed target computational budgets despite its importance to resource-bounded devices. We present Recursive Speculative Decoding (RSD), a novel tree-based method that samples draft tokens without replacement and maximizes the diversity of the tree. During RSD's drafting, the tree is built by either Gumbel-Top-$k$ trick that draws tokens without replacement in parallel or Stochastic Beam Search that samples sequences without replacement while early-truncating unlikely draft sequences and reducing the computational cost of LLM. We empirically evaluate RSD with Llama 2 and OPT models, showing that RSD outperforms the baseline methods, consistently for fixed draft sequence length and in most cases for fixed computational budgets at LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02574v1">ChatCite: LLM Agent with Human Workflow Guidance for Comparative Literature Summary</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 18 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The literature review is an indispensable step in the research process. It provides the benefit of comprehending the research problem and understanding the current research situation while conducting a comparative analysis of prior works. However, literature summary is challenging and time consuming. The previous LLM-based studies on literature review mainly focused on the complete process, including literature retrieval, screening, and summarization. However, for the summarization step, simple CoT method often lacks the ability to provide extensive comparative summary. In this work, we firstly focus on the independent literature summarization step and introduce ChatCite, an LLM agent with human workflow guidance for comparative literature summary. This agent, by mimicking the human workflow, first extracts key elements from relevant literature and then generates summaries using a Reflective Incremental Mechanism. In order to better evaluate the quality of the generated summaries, we devised a LLM-based automatic evaluation metric, G-Score, in refer to the human evaluation criteria. The ChatCite agent outperformed other models in various dimensions in the experiments. The literature summaries generated by ChatCite can also be directly used for drafting literature reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13720v2">Can LLMs Fix Issues with Reasoning Models? Towards More Likely Models for AI Planning</a></div>
    <div class="paper-meta">
      📅 2024-03-05
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      This is the first work to look at the application of large language models (LLMs) for the purpose of model space edits in automated planning tasks. To set the stage for this union, we explore two different flavors of model space problems that have been studied in the AI planning literature and explore the effect of an LLM on those tasks. We empirically demonstrate how the performance of an LLM contrasts with combinatorial search (CS) -- an approach that has been traditionally used to solve model space tasks in planning, both with the LLM in the role of a standalone model space reasoner as well as in the role of a statistical signal in concert with the CS approach as part of a two-stage process. Our experiments show promising results suggesting further forays of LLMs into the exciting world of model space reasoning for planning tasks in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02513v1">Balancing Enhancement, Harmlessness, and General Capabilities: Enhancing Conversational LLMs with Direct RLHF</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      In recent advancements in Conversational Large Language Models (LLMs), a concerning trend has emerged, showing that many new base LLMs experience a knowledge reduction in their foundational capabilities following Supervised Fine-Tuning (SFT). This process often leads to issues such as forgetting or a decrease in the base model's abilities. Moreover, fine-tuned models struggle to align with user preferences, inadvertently increasing the generation of toxic outputs when specifically prompted. To overcome these challenges, we adopted an innovative approach by completely bypassing SFT and directly implementing Harmless Reinforcement Learning from Human Feedback (RLHF). Our method not only preserves the base model's general capabilities but also significantly enhances its conversational abilities, while notably reducing the generation of toxic outputs. Our approach holds significant implications for fields that demand a nuanced understanding and generation of responses, such as customer service. We applied this methodology to Mistral, the most popular base model, thereby creating Mistral-Plus. Our validation across 11 general tasks demonstrates that Mistral-Plus outperforms similarly sized open-source base models and their corresponding instruct versions. Importantly, the conversational abilities of Mistral-Plus were significantly improved, indicating a substantial advancement over traditional SFT models in both safety and user preference alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11980v5">LLM-based Smart Reply (LSR): Enhancing Collaborative Performance with ChatGPT-mediated Smart Reply System</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      Interactive user interfaces have increasingly explored AI's role in enhancing communication efficiency and productivity in collaborative tasks. The emergence of Large Language Models (LLMs) such as ChatGPT has revolutionized conversational agents, employing advanced deep learning techniques to generate context-aware, coherent, and personalized responses. Consequently, LLM-based AI assistants provide a more natural and efficient user experience across various scenarios. In this paper, we study how LLM models can be used to improve work efficiency in collaborative workplaces. Specifically, we present an LLM-based Smart Reply (LSR) system utilizing the ChatGPT to generate personalized responses in professional collaborative scenarios while adapting to context and communication style based on prior responses. Our two-step process involves generating a preliminary response type (e.g., Agree, Disagree) to provide a generalized direction for message generation, thus reducing response drafting time. We conducted an experiment where participants completed simulated work tasks involving a Dual N-back test and subtask scheduling through Google Calendar while interacting with co-workers. Our findings indicate that the proposed LSR reduces overall workload, as measured by the NASA TLX, and improves work performance and productivity in the N-back task. We also provide qualitative analysis based on participants' experiences, as well as design considerations to provide future directions for improving such implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02475v1">Enhancing LLM Safety via Constrained Direct Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      The rapidly increasing capabilities of large language models (LLMs) raise an urgent need to align AI systems with diverse human preferences to simultaneously enhance their usefulness and safety, despite the often conflicting nature of these goals. To address this important problem, a promising approach is to enforce a safety constraint at the fine-tuning stage through a constrained Reinforcement Learning from Human Feedback (RLHF) framework. This approach, however, is computationally expensive and often unstable. In this work, we introduce Constrained DPO (C-DPO), a novel extension of the recently proposed Direct Preference Optimization (DPO) approach for fine-tuning LLMs that is both efficient and lightweight. By integrating dual gradient descent and DPO, our method identifies a nearly optimal trade-off between helpfulness and harmlessness without using reinforcement learning. Empirically, our approach provides a safety guarantee to LLMs that is missing in DPO while achieving significantly higher rewards under the same safety constraint compared to a recently proposed safe RLHF approach. Warning: This paper contains example data that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.01264v2">Exploring the psychology of LLMs' Moral and Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit expert-level performance in tasks across a wide range of different domains. Ethical issues raised by LLMs and the need to align future versions makes it important to know how state of the art models reason about moral and legal issues. In this paper, we employ the methods of experimental psychology to probe into this question. We replicate eight studies from the experimental literature with instances of Google's Gemini Pro, Anthropic's Claude 2.1, OpenAI's GPT-4, and Meta's Llama 2 Chat 70b. We find that alignment with human responses shifts from one experiment to another, and that models differ amongst themselves as to their overall alignment, with GPT-4 taking a clear lead over all other models we tested. Nonetheless, even when LLM-generated responses are highly correlated to human responses, there are still systematic differences, with a tendency for models to exaggerate effects that are present among humans, in part by reducing variance. This recommends caution with regards to proposals of replacing human participants with current state-of-the-art LLMs in psychological research and highlights the need for further research about the distinctive aspects of machine psychology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13655v3">LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 Transactions on Machine Learning Research (TMLR) 2024, with Featured Certification
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-image diffusion models have yielded impressive results in generating realistic and diverse images. However, these models still struggle with complex prompts, such as those that involve numeracy and spatial reasoning. This work proposes to enhance prompt understanding capabilities in diffusion models. Our method leverages a pretrained large language model (LLM) for grounded generation in a novel two-stage process. In the first stage, the LLM generates a scene layout that comprises captioned bounding boxes from a given prompt describing the desired image. In the second stage, a novel controller guides an off-the-shelf diffusion model for layout-grounded image generation. Both stages utilize existing pretrained models without additional model parameter optimization. Our method significantly outperforms the base diffusion model and several strong baselines in accurately generating images according to prompts that require various capabilities, doubling the generation accuracy across four tasks on average. Furthermore, our method enables instruction-based multi-round scene specification and can handle prompts in languages not supported by the underlying diffusion model. We anticipate that our method will unleash users' creativity by accurately following more complex prompts. Our code, demo, and benchmark are available at: https://llm-grounded-diffusion.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18535v1">DRAK: Unlocking Molecular Insights with Domain-Specific Retrieval-Augmented Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 Ongoing work; 11 pages, 6 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encounter challenges with the unique syntax of specific domains, such as biomolecules. Existing fine-tuning or modality alignment techniques struggle to bridge the domain knowledge gap and understand complex molecular data, limiting LLMs' progress in specialized fields. To overcome these limitations, we propose an expandable and adaptable non-parametric knowledge injection framework named Domain-specific Retrieval-Augmented Knowledge (DRAK), aimed at enhancing reasoning capabilities in specific domains. Utilizing knowledge-aware prompts and gold label-induced reasoning, DRAK has developed profound expertise in the molecular domain and the capability to handle a broad spectrum of analysis tasks. We evaluated two distinct forms of DRAK variants, proving that DRAK exceeds previous benchmarks on six molecular tasks within the Mol-Instructions dataset. Extensive experiments have underscored DRAK's formidable performance and its potential to unlock molecular insights, offering a unified paradigm for LLMs to tackle knowledge-intensive tasks in specific domains. Our code will be available soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01999v1">LLM-Oriented Retrieval Tuner</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 16 pages, 8 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Dense Retrieval (DR) is now considered as a promising tool to enhance the memorization capacity of Large Language Models (LLM) such as GPT3 and GPT-4 by incorporating external memories. However, due to the paradigm discrepancy between text generation of LLM and DR, it is still an open challenge to integrate the retrieval and generation tasks in a shared LLM. In this paper, we propose an efficient LLM-Oriented Retrieval Tuner, namely LMORT, which decouples DR capacity from base LLM and non-invasively coordinates the optimally aligned and uniform layers of the LLM towards a unified DR space, achieving an efficient and effective DR without tuning the LLM itself. The extensive experiments on six BEIR datasets show that our approach could achieve competitive zero-shot retrieval performance compared to a range of strong DR models while maintaining the generation ability of LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01876v1">DéjàVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      Distributed LLM serving is costly and often underutilizes hardware accelerators due to three key challenges: bubbles in pipeline-parallel deployments caused by the bimodal latency of prompt and token processing, GPU memory overprovisioning, and long recovery times in case of failures. In this paper, we propose D\'ej\`aVu, a system to address all these challenges using a versatile and efficient KV cache streaming library (D\'ej\`aVuLib). Using D\'ej\`aVuLib, we propose and implement efficient prompt-token disaggregation to reduce pipeline bubbles, microbatch swapping for efficient GPU memory management, and state replication for fault-tolerance. We highlight the efficacy of these solutions on a range of large models across cloud deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01784v1">CatCode: A Comprehensive Evaluation Framework for LLMs On the Mixture of Code and Text</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) such as ChatGPT are increasingly proficient in understanding and generating a mixture of code and text. Evaluation based on such $\textit{mixture}$ can lead to a more comprehensive understanding of the models' abilities in solving coding problems. However, in this context, current evaluation methods are either limited in task coverage or lack standardization. To address this issue, we propose using category theory as a framework for evaluation. Specifically, morphisms within a code category can represent code debugging and transformation, functors between two categories represent code translation, and functors between a code category and a natural language category represent code generation, explanation, and reproduction. We present an automatic evaluation framework called $\textbf{CatCode}$ ($\textbf{Cat}$egory $\textbf{Code}$) that can comprehensively assess the coding abilities of LLMs, including ChatGPT, Text-Davinci, and CodeGeeX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01757v1">How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 8pages,3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have notably positioned them as capable tools for addressing complex optimization challenges. Despite this recognition, a predominant limitation of existing LLM-based optimization methods is their struggle to capture the relationships among decision variables when relying exclusively on numerical text prompts, especially in high-dimensional problems. Keeping this in mind, we first propose to enhance the optimization performance using multimodal LLM capable of processing both textual and visual prompts for deeper insights of the processed optimization problem. This integration allows for a more comprehensive understanding of optimization problems, akin to human cognitive processes. We have developed a multimodal LLM-based optimization framework that simulates human problem-solving workflows, thereby offering a more nuanced and effective analysis. The efficacy of this method is evaluated through extensive empirical studies focused on a well-known combinatorial optimization problem, i.e., capacitated vehicle routing problem. The results are compared against those obtained from the LLM-based optimization algorithms that rely solely on textual prompts, demonstrating the significant advantages of our multimodal approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17256v2">Beyond the Known: Investigating LLMs Performance on Out-of-Domain Intent Detection</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      Out-of-domain (OOD) intent detection aims to examine whether the user's query falls outside the predefined domain of the system, which is crucial for the proper functioning of task-oriented dialogue (TOD) systems. Previous methods address it by fine-tuning discriminative models. Recently, some studies have been exploring the application of large language models (LLMs) represented by ChatGPT to various downstream tasks, but it is still unclear for their ability on OOD detection task.This paper conducts a comprehensive evaluation of LLMs under various experimental settings, and then outline the strengths and weaknesses of LLMs. We find that LLMs exhibit strong zero-shot and few-shot capabilities, but is still at a disadvantage compared to models fine-tuned with full resource. More deeply, through a series of additional analysis experiments, we discuss and summarize the challenges faced by LLMs and provide guidance for future work including injecting domain knowledge, strengthening knowledge transfer from IND(In-domain) to OOD, and understanding long instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15727v2">LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 Fixed the bibliography reference issue in our LLM jailbreak defense vision paper submitted on 24 Feb 2024
    </div>
    <details class="paper-abstract">
      Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01709v1">Can LLMs Generate Architectural Design Decisions? -An Exploratory Empirical study</a></div>
    <div class="paper-meta">
      📅 2024-03-04
      | 💬 This paper has been accepted to IEEE ICSA 2024 (Main Track - Research Track)
    </div>
    <details class="paper-abstract">
      Architectural Knowledge Management (AKM) involves the organized handling of information related to architectural decisions and design within a project or organization. An essential artifact of AKM is the Architecture Decision Records (ADR), which documents key design decisions. ADRs are documents that capture decision context, decision made and various aspects related to a design decision, thereby promoting transparency, collaboration, and understanding. Despite their benefits, ADR adoption in software development has been slow due to challenges like time constraints and inconsistent uptake. Recent advancements in Large Language Models (LLMs) may help bridge this adoption gap by facilitating ADR generation. However, the effectiveness of LLM for ADR generation or understanding is something that has not been explored. To this end, in this work, we perform an exploratory study that aims to investigate the feasibility of using LLM for the generation of ADRs given the decision context. In our exploratory study, we utilize GPT and T5-based models with 0-shot, few-shot, and fine-tuning approaches to generate the Decision of an ADR given its Context. Our results indicate that in a 0-shot setting, state-of-the-art models such as GPT-4 generate relevant and accurate Design Decisions, although they fall short of human-level performance. Additionally, we observe that more cost-effective models like GPT-3.5 can achieve similar outcomes in a few-shot setting, and smaller models such as Flan-T5 can yield comparable results after fine-tuning. To conclude, this exploratory study suggests that LLM can generate Design Decisions, but further research is required to attain human-level generation and establish standardized widespread adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16063v3">Citation-Enhanced Generation for LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-03-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit powerful general intelligence across diverse scenarios, including their integration into chatbots. However, a vital challenge of LLM-based chatbots is that they may produce hallucinated content in responses, which significantly limits their applicability. Various efforts have been made to alleviate hallucination, such as retrieval augmented generation and reinforcement learning with human feedback, but most of them require additional training and data annotation. In this paper, we propose a novel post-hoc Citation-Enhanced Generation (CEG) approach combined with retrieval argumentation. Unlike previous studies that focus on preventing hallucinations during generation, our method addresses this issue in a post-hoc way. It incorporates a retrieval module to search for supporting documents relevant to the generated content, and employs a natural language inference-based citation generation module. Once the statements in the generated content lack of reference, our model can regenerate responses until all statements are supported by citations. Note that our method is a training-free plug-and-play plugin that is capable of various LLMs. Experiments on various hallucination-related datasets show our framework outperforms state-of-the-art methods in both hallucination detection and response regeneration on three benchmarks. Our codes and dataset will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01626v1">Using LLMs for Tabletop Exercises within the Security Domain</a></div>
    <div class="paper-meta">
      📅 2024-03-03
      | 💬 7 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tabletop exercises are a crucial component of many company's strategy to test and evaluate its preparedness for security incidents in a realistic way. Traditionally led by external firms specializing in cybersecurity, these exercises can be costly, time-consuming, and may not always align precisely with the client's specific needs. Large Language Models (LLMs) like ChatGPT offer a compelling alternative. They enable faster iteration, provide rich and adaptable simulations, and offer infinite patience in handling feedback and recommendations. This approach can enhances the efficiency and relevance of security preparedness exercises.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16119v3">Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition</a></div>
    <div class="paper-meta">
      📅 2024-03-03
      | 💬 34 pages, 8 figures Codebase: https://github.com/PromptLabs/hackaprompt Dataset: https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md Playground: https://huggingface.co/spaces/hackaprompt/playground
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05583v1">A Cross-Modal Approach to Silent Speech with LLM-Enhanced Recognition</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      Silent Speech Interfaces (SSIs) offer a noninvasive alternative to brain-computer interfaces for soundless verbal communication. We introduce Multimodal Orofacial Neural Audio (MONA), a system that leverages cross-modal alignment through novel loss functions--cross-contrast (crossCon) and supervised temporal contrast (supTcon)--to train a multimodal model with a shared latent representation. This architecture enables the use of audio-only datasets like LibriSpeech to improve silent speech recognition. Additionally, our introduction of Large Language Model (LLM) Integrated Scoring Adjustment (LISA) significantly improves recognition accuracy. Together, MONA LISA reduces the state-of-the-art word error rate (WER) from 28.8% to 12.2% in the Gaddy (2020) benchmark dataset for silent speech on an open vocabulary. For vocal EMG recordings, our method improves the state-of-the-art from 23.3% to 3.7% WER. In the Brain-to-Text 2024 competition, LISA performs best, improving the top WER from 9.8% to 8.9%. To the best of our knowledge, this work represents the first instance where noninvasive silent speech recognition on an open vocabulary has cleared the threshold of 15% WER, demonstrating that SSIs can be a viable alternative to automatic speech recognition (ASR). Our work not only narrows the performance gap between silent and vocalized speech but also opens new possibilities in human-computer interaction, demonstrating the potential of cross-modal approaches in noisy and data-limited regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01273v1">NoMAD-Attention: Efficient LLM Inference on CPUs Through Multiply-add-free Attention</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      Large language model inference on Central Processing Units (CPU) is challenging due to the vast quantities of expensive Multiply-Add (MAD) matrix operations in the attention computations. In this paper, we argue that there is a rare gem in modern CPUs, Single-Instruction-Multiple-Data (SIMD) registers, which allow for ultra-low-latency lookups in batch. We leverage this unique capability of CPUs to propose NoMAD-Attention, an efficient attention algorithm that replaces MAD operations with in-register lookups. Through hardware-aware algorithmic designs, NoMAD-Attention achieves the computation of attention scores using repeated fast accesses to SIMD registers despite their highly limited sizes. Moreover, NoMAD-Attention works with pre-trained attention-based LLMs without model finetuning. Empirical evaluations demonstrate that NoMAD-Attention maintains the quality of the original LLMs well, and speeds up the 4-bit quantized LLaMA-7B-based model by up to 2$\times$ at 16k context length. Our results are reproducible at https://github.com/tonyzhang617/nomad-dist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01271v1">Employing LLMs for Incident Response Planning and Review</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 10 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Incident Response Planning (IRP) is essential for effective cybersecurity management, requiring detailed documentation (or playbooks) to guide security personnel during incidents. Yet, creating comprehensive IRPs is often hindered by challenges such as complex systems, high turnover rates, and legacy technologies lacking documentation. This paper argues that, despite these obstacles, the development, review, and refinement of IRPs can be significantly enhanced through the utilization of Large Language Models (LLMs) like ChatGPT. By leveraging LLMs for tasks such as drafting initial plans, suggesting best practices, and identifying documentation gaps, organizations can overcome resource constraints and improve their readiness for cybersecurity incidents. We discuss the potential of LLMs to streamline IRP processes, while also considering the limitations and the need for human oversight in ensuring the accuracy and relevance of generated content. Our findings contribute to the cybersecurity field by demonstrating a novel approach to enhancing IRP with AI technologies, offering practical insights for organizations seeking to bolster their incident response capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01248v1">SceneCraft: An LLM Agent for Synthesizing 3D Scene as Blender Code</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      This paper introduces SceneCraft, a Large Language Model (LLM) Agent converting text descriptions into Blender-executable Python scripts which render complex scenes with up to a hundred 3D assets. This process requires complex spatial planning and arrangement. We tackle these challenges through a combination of advanced abstraction, strategic planning, and library learning. SceneCraft first models a scene graph as a blueprint, detailing the spatial relationships among assets in the scene. SceneCraft then writes Python scripts based on this graph, translating relationships into numerical constraints for asset layout. Next, SceneCraft leverages the perceptual strengths of vision-language foundation models like GPT-V to analyze rendered images and iteratively refine the scene. On top of this process, SceneCraft features a library learning mechanism that compiles common script functions into a reusable library, facilitating continuous self-improvement without expensive LLM parameter tuning. Our evaluation demonstrates that SceneCraft surpasses existing LLM-based agents in rendering complex scenes, as shown by its adherence to constraints and favorable human assessments. We also showcase the broader application potential of SceneCraft by reconstructing detailed 3D scenes from the Sintel movie and guiding a video generative model with generated scenes as intermediary control signal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17411v2">Consistency Matters: Explore LLMs Consistency From a Black-Box Perspective</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 This paper is not ready
    </div>
    <details class="paper-abstract">
      Nowadays both commercial and open-source academic LLM have become the mainstream models of NLP. However, there is still a lack of research on LLM consistency, meaning that throughout the various stages of LLM research and deployment, its internal parameters and capabilities should remain unchanged. This issue exists in both the industrial and academic sectors. The solution to this problem is often time-consuming and labor-intensive, and there is also an additional cost of secondary deployment, resulting in economic and time losses. To fill this gap, we build an LLM consistency task dataset and design several baselines. Additionally, we choose models of diverse scales for the main experiments. Specifically, in the LightGBM experiment, we used traditional NLG metrics (i.e., ROUGE, BLEU, METEOR) as the features needed for model training. The final result exceeds the manual evaluation and GPT3.5 as well as other models in the main experiment, achieving the best performance. In the end, we use the best performing LightGBM model as the base model to build the evaluation tool, which can effectively assist in the deployment of business models. Our code and tool demo are available at https://github.com/heavenhellchen/Consistency.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11094v2">Word Embeddings Revisited: Do LLMs Offer Something New?</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Learning meaningful word embeddings is key to training a robust language model. The recent rise of Large Language Models (LLMs) has provided us with many new word/sentence/document embedding models. Although LLMs have shown remarkable advancement in various NLP tasks, it is still unclear whether the performance improvement is merely because of scale or whether underlying embeddings they produce significantly differ from classical encoding models like Sentence-BERT (SBERT) or Universal Sentence Encoder (USE). This paper systematically investigates this issue by comparing classical word embedding techniques against LLM-based word embeddings in terms of their latent vector semantics. Our results show that LLMs tend to cluster semantically related words more tightly than classical models. LLMs also yield higher average accuracy on the Bigger Analogy Test Set (BATS) over classical methods. Finally, some LLMs tend to produce word embeddings similar to SBERT, a relatively lighter classical model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01209v1">Data-free Multi-label Image Recognition via LLM-powered Prompt Tuning</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      This paper proposes a novel framework for multi-label image recognition without any training data, called data-free framework, which uses knowledge of pre-trained Large Language Model (LLM) to learn prompts to adapt pretrained Vision-Language Model (VLM) like CLIP to multilabel classification. Through asking LLM by well-designed questions, we acquire comprehensive knowledge about characteristics and contexts of objects, which provides valuable text descriptions for learning prompts. Then we propose a hierarchical prompt learning method by taking the multi-label dependency into consideration, wherein a subset of category-specific prompt tokens are shared when the corresponding objects exhibit similar attributes or are more likely to co-occur. Benefiting from the remarkable alignment between visual and linguistic semantics of CLIP, the hierarchical prompts learned from text descriptions are applied to perform classification of images during inference. Our framework presents a new way to explore the synergies between multiple pre-trained models for novel category recognition. Extensive experiments on three public datasets (MS-COCO, VOC2007, and NUS-WIDE) demonstrate that our method achieves better results than the state-of-the-art methods, especially outperforming the zero-shot multi-label recognition methods by 4.7% in mAP on MS-COCO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01185v1">Balancing Exploration and Exploitation in LLM using Soft RLLF for Enhanced Negation Understanding</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 JURISIN 2024
    </div>
    <details class="paper-abstract">
      Finetuning approaches in NLP often focus on exploitation rather than exploration, which may lead to suboptimal models. Given the vast search space of natural language, this limited exploration can restrict their performance in complex, high-stakes domains, where accurate negation understanding and logical reasoning abilities are crucial. To address this issue, we leverage Reinforcement Learning from Logical Feedback (RLLF) to create an effective balance between exploration and exploitation in LLMs. Our approach employs an appropriate benchmark dataset for training and evaluation, highlighting the importance of exploration in enhancing negation understanding capabilities. We compare the performance of our RLLF-enhanced LLMs with baseline models trained without RLLF, demonstrating the value of this balanced approach. Furthermore, we showcase the potential of our method in legal AI applications by employing transfer learning and evaluating its impact on negation understanding. Our experimental results exhibit the effectiveness of balancing exploration and exploitation with RLLF in improving LLMs' negation capabilities. This has implications for the development of more accurate, reliable, and logically consistent language models in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.05499v2">Prompt Injection attack against LLM-integrated Applications</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and severe attack outcomes, such as unrestricted arbitrary LLM usage and uncomplicated application prompt theft. We deploy HouYi on 36 actual LLM-integrated applications and discern 31 applications susceptible to prompt injection. 10 vendors have validated our discoveries, including Notion, which has the potential to impact millions of users. Our investigation illuminates both the possible risks of prompt injection attacks and the possible tactics for mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01136v1">LLM-PQ: Serving LLM on Heterogeneous Clusters with Phase-Aware Partition and Adaptive Quantization</a></div>
    <div class="paper-meta">
      📅 2024-03-02
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in Large-scale language models (LLMs) have demonstrated impressive performance on various tasks. The immense sizes of LLMs have led to very high resource demand and cost for running the models. Though the models are largely served using uniform high-caliber GPUs nowadays, utilizing a heterogeneous cluster with a mix of available high- and low-capacity GPUs can potentially substantially reduce the serving cost. There is a lack of designs to support efficient LLM serving using a heterogeneous cluster, while the current solutions focus on model partition and uniform compression among homogeneous devices. This paper proposes LLM-PQ, a system that advocates adaptive model quantization and phase-aware partition to improve LLM serving efficiency on heterogeneous GPU clusters. We carefully decide on mixed-precision model quantization together with phase-aware model partition and micro-batch sizing in distributed LLM serving with an efficient algorithm, to greatly enhance inference throughput while fulfilling user-specified model quality targets. Extensive experiments on production inference workloads in 11 different clusters demonstrate that LLM-PQ achieves up to 2.88x (2.26x on average) throughput improvement in inference, showing great advantages over state-of-the-art works.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.19248v2">Let LLMs Take on the Latest Challenges! A Chinese Dynamic Question Answering Benchmark</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 Work in progress!
    </div>
    <details class="paper-abstract">
      How to better evaluate the capabilities of Large Language Models (LLMs) is the focal point and hot topic in current LLMs research. Previous work has noted that due to the extremely high cost of iterative updates of LLMs, they are often unable to answer the latest dynamic questions well. To promote the improvement of Chinese LLMs' ability to answer dynamic questions, in this paper, we introduce CDQA, a Chinese Dynamic QA benchmark containing question-answer pairs related to the latest news on the Chinese Internet. We obtain high-quality data through a pipeline that combines humans and models, and carefully classify the samples according to the frequency of answer changes to facilitate a more fine-grained observation of LLMs' capabilities. We have also evaluated and analyzed mainstream and advanced Chinese LLMs on CDQA. Extensive experiments and valuable insights suggest that our proposed CDQA is challenging and worthy of more further study. We believe that the benchmark we provide will become one of the key data resources for improving LLMs' Chinese question-answering ability in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.15337v3">Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation</a></div>
    <div class="paper-meta">
      📅 2024-03-02
      | 💬 In ICLR'24
    </div>
    <details class="paper-abstract">
      This work aims at decreasing the end-to-end generation latency of large language models (LLMs). One of the major causes of the high generation latency is the sequential decoding approach adopted by almost all state-of-the-art LLMs. In this work, motivated by the thinking and writing process of humans, we propose Skeleton-of-Thought (SoT), which first guides LLMs to generate the skeleton of the answer, and then conducts parallel API calls or batched decoding to complete the contents of each skeleton point in parallel. Not only does SoT provide considerable speed-ups across 12 LLMs, but it can also potentially improve the answer quality on several question categories. SoT is an initial attempt at data-centric optimization for inference efficiency, and showcases the potential of eliciting high-quality answers by explicitly planning the answer structure in language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.01008v1">BasedAI: A decentralized P2P network for Zero Knowledge Large Language Models (ZK-LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-03-01
    </div>
    <details class="paper-abstract">
      BasedAI is a distributed network of machines which introduces decentralized infrastructure capable of integrating Fully Homomorphic Encryption (FHE) with any large language model (LLM) connected to its network. The proposed framework embeds a default mechanism, called "Cerberus Squeezing", into the mining process which enables the transformation of a standard LLMs into encrypted zero-knowledge LLMs, or "ZK-LLMs", leveraging insights from generative adversarial networks for data privacy. This novel quantization mechanism empowers BasedAI miners to process and respond to prompts derived from User interaction with LLMs without the need for decrypting either the queries or their corresponding responses. The introduction of Cerberus Squeezing significantly improves performance degradation caused by quantized functions in current FHE-compliant computing environments by proactively optimizing calls between users, miners, and validators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00990v1">Formulation Comparison for Timeline Construction using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-03-01
    </div>
    <details class="paper-abstract">
      Constructing a timeline requires identifying the chronological order of events in an article. In prior timeline construction datasets, temporal orders are typically annotated by either event-to-time anchoring or event-to-event pairwise ordering, both of which suffer from missing temporal information. To mitigate the issue, we develop a new evaluation dataset, TimeSET, consisting of single-document timelines with document-level order annotation. TimeSET features saliency-based event selection and partial ordering, which enable a practical annotation workload. Aiming to build better automatic timeline construction systems, we propose a novel evaluation framework to compare multiple task formulations with TimeSET by prompting open LLMs, i.e., Llama 2 and Flan-T5. Considering that identifying temporal orders of events is a core subtask in timeline construction, we further benchmark open LLMs on existing event temporal ordering datasets to gain a robust understanding of their capabilities. Our experiments show that (1) NLI formulation with Flan-T5 demonstrates a strong performance among others, while (2) timeline construction and event temporal ordering are still challenging tasks for few-shot LLMs. Our code and data are available at https://github.com/kimihiroh/timeset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00964v1">MALTO at SemEval-2024 Task 6: Leveraging Synthetic Data for LLM Hallucination Detection</a></div>
    <div class="paper-meta">
      📅 2024-03-01
      | 💬 Under revision at SemEval 2024
    </div>
    <details class="paper-abstract">
      In Natural Language Generation (NLG), contemporary Large Language Models (LLMs) face several challenges, such as generating fluent yet inaccurate outputs and reliance on fluency-centric metrics. This often leads to neural networks exhibiting "hallucinations". The SHROOM challenge focuses on automatically identifying these hallucinations in the generated text. To tackle these issues, we introduce two key components, a data augmentation pipeline incorporating LLM-assisted pseudo-labelling and sentence rephrasing, and a voting ensemble from three models pre-trained on Natural Language Inference (NLI) tasks and fine-tuned on diverse datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00745v1">AtP*: An efficient and scalable method for localizing LLM behaviour to components</a></div>
    <div class="paper-meta">
      📅 2024-03-01
    </div>
    <details class="paper-abstract">
      Activation Patching is a method of directly computing causal attributions of behavior to model components. However, applying it exhaustively requires a sweep with cost scaling linearly in the number of model components, which can be prohibitively expensive for SoTA Large Language Models (LLMs). We investigate Attribution Patching (AtP), a fast gradient-based approximation to Activation Patching and find two classes of failure modes of AtP which lead to significant false negatives. We propose a variant of AtP called AtP*, with two changes to address these failure modes while retaining scalability. We present the first systematic study of AtP and alternative methods for faster activation patching and show that AtP significantly outperforms all other investigated methods, with AtP* providing further significant improvement. Finally, we provide a method to bound the probability of remaining false negatives of AtP* estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00690v1">Playing NetHack with LLMs: Potential & Limitations as Zero-Shot Agents</a></div>
    <div class="paper-meta">
      📅 2024-03-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great success as high-level planners for zero-shot game-playing agents. However, these agents are primarily evaluated on Minecraft, where long-term planning is relatively straightforward. In contrast, agents tested in dynamic robot environments face limitations due to simplistic environments with only a few objects and interactions. To fill this gap in the literature, we present NetPlay, the first LLM-powered zero-shot agent for the challenging roguelike NetHack. NetHack is a particularly challenging environment due to its diverse set of items and monsters, complex interactions, and many ways to die. NetPlay uses an architecture designed for dynamic robot environments, modified for NetHack. Like previous approaches, it prompts the LLM to choose from predefined skills and tracks past interactions to enhance decision-making. Given NetHack's unpredictable nature, NetPlay detects important game events to interrupt running skills, enabling it to react to unforeseen circumstances. While NetPlay demonstrates considerable flexibility and proficiency in interacting with NetHack's mechanics, it struggles with ambiguous task descriptions and a lack of explicit feedback. Our findings demonstrate that NetPlay performs best with detailed context information, indicating the necessity for dynamic methods in supplying context information for complex games such as NetHack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14270v2">Take the Bull by the Horns: Hard Sample-Reweighted Continual Training Improves LLM Generalization</a></div>
    <div class="paper-meta">
      📅 2024-03-01
      | 💬 Preprint; updated reference and related works
    </div>
    <details class="paper-abstract">
      In the rapidly advancing arena of large language models (LLMs), a key challenge is to enhance their capabilities amid a looming shortage of high-quality training data. Our study starts from an empirical strategy for the light continual training of LLMs using their original pre-training data sets, with a specific focus on selective retention of samples that incur moderately high losses. These samples are deemed informative and beneficial for model refinement, contrasting with the highest-loss samples, which would be discarded due to their correlation with data noise and complexity. We then formalize this strategy into a principled framework of Instance-Reweighted Distributionally Robust Optimization (IR-DRO). IR-DRO is designed to dynamically prioritize the training focus on informative samples through an instance reweighting mechanism, streamlined by a closed-form solution for straightforward integration into established training protocols. Through rigorous experimentation with various models and datasets, our findings indicate that our sample-targeted methods significantly improve LLM performance across multiple benchmarks, in both continual pre-training and instruction tuning scenarios. Our codes are available at https://github.com/VITA-Group/HardFocusTraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08107v2">SAIE Framework: Support Alone Isn't Enough -- Advancing LLM Training with Adversarial Remarks</a></div>
    <div class="paper-meta">
      📅 2024-03-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can justify or critique their predictions through discussions with other models or humans, thereby enriching their intrinsic understanding of instances. While proactive discussions in the inference phase have been shown to boost performance, such interactions have not been extensively explored during the training phase. We hypothesize that incorporating interactive discussions into the training process can enhance the models' understanding and improve their reasoning and verbal expression abilities during inference. This work introduces the SAIE framework, which facilitates supportive and adversarial discussions between learner and partner models. The learner model receives responses from the partner, and its parameters are then updated based on this discussion. This dynamic adjustment process continues throughout the training phase, responding to the evolving outputs of the learner model. Our empirical evaluation across various tasks, including math problems, commonsense reasoning, and multi-domain knowledge, demonstrates that models fine-tuned with the SAIE framework outperform those trained with conventional fine-tuning approaches. Furthermore, our method enhances the models' reasoning capabilities, improving both individual and multi-agent inference performance.
    </details>
</div>
