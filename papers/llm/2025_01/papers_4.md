# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09164v1">The Veln(ia)s is in the Details: Evaluating LLM Judgment on Latvian and Lithuanian Short Answer Matching</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      In this work, we address the challenge of evaluating large language models (LLMs) on the short answer matching task for Latvian and Lithuanian languages. We introduce novel datasets consisting of 502 Latvian and 690 Lithuanian question-answer pairs. For each question-answer pair, we generated matched and non-matched answers using a set of alteration rules specifically designed to introduce small but meaningful changes in the text. These generated answers serve as test cases to assess the ability of LLMs to detect subtle differences in matching of the original answers. A subset of the datasets was manually verified for quality and accuracy. Our results show that while larger LLMs, such as QWEN2.5 72b and LLaMa3.1 70b, demonstrate near-perfect performance in distinguishing matched and non-matched answers, smaller models show more variance. For instance, LLaMa3.1 8b and EuroLLM 9b benefited from few-shot examples, while Mistral Nemo 12b underperformed on detection of subtle text alteration, particularly in Lithuanian, even with additional examples. QWEN2.5 7b and Mistral 7b were able to obtain a strong and comparable performance to the larger 70b models in zero and few shot experiments. Moreover, the performance of Mistral 7b was weaker in few shot experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09154v1">Towards Multilingual LLM Evaluation for Baltic and Nordic languages: A study on Lithuanian History</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      In this work, we evaluated Lithuanian and general history knowledge of multilingual Large Language Models (LLMs) on a multiple-choice question-answering task. The models were tested on a dataset of Lithuanian national and general history questions translated into Baltic, Nordic, and other languages (English, Ukrainian, Arabic) to assess the knowledge sharing from culturally and historically connected groups. We evaluated GPT-4o, LLaMa3.1 8b and 70b, QWEN2.5 7b and 72b, Mistral Nemo 12b, LLaMa3 8b, Mistral 7b, LLaMa3.2 3b, and Nordic fine-tuned models (GPT-SW3 and LLaMa3 8b). Our results show that GPT-4o consistently outperformed all other models across language groups, with slightly better results for Baltic and Nordic languages. Larger open-source models like QWEN2.5 72b and LLaMa3.1 70b performed well but showed weaker alignment with Baltic languages. Smaller models (Mistral Nemo 12b, LLaMa3.2 3b, QWEN 7B, LLaMa3.1 8B, and LLaMa3 8b) demonstrated gaps with LT-related alignment with Baltic languages while performing better on Nordic and other languages. The Nordic fine-tuned models did not surpass multilingual models, indicating that shared cultural or historical context alone does not guarantee better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09127v1">Multilingual LLMs Struggle to Link Orthography and Semantics in Bilingual Word Processing</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 Code available at: https://github.com/EshaanT/Bilingual_processing_LLMs
    </div>
    <details class="paper-abstract">
      Bilingual lexical processing is shaped by the complex interplay of phonological, orthographic, and semantic features of two languages within an integrated mental lexicon. In humans, this is evident in the ease with which cognate words - words similar in both orthographic form and meaning (e.g., blind, meaning "sightless" in both English and German) - are processed, compared to the challenges posed by interlingual homographs, which share orthographic form but differ in meaning (e.g., gift, meaning "present" in English but "poison" in German). We investigate how multilingual Large Language Models (LLMs) handle such phenomena, focusing on English-Spanish, English-French, and English-German cognates, non-cognate, and interlingual homographs. Specifically, we evaluate their ability to disambiguate meanings and make semantic judgments, both when these word types are presented in isolation or within sentence contexts. Our findings reveal that while certain LLMs demonstrate strong performance in recognizing cognates and non-cognates in isolation, they exhibit significant difficulty in disambiguating interlingual homographs, often performing below random baselines. This suggests LLMs tend to rely heavily on orthographic similarities rather than semantic understanding when interpreting interlingual homographs. Further, we find LLMs exhibit difficulty in retrieving word meanings, with performance in isolative disambiguation tasks having no correlation with semantic understanding. Finally, we study how the LLM processes interlingual homographs in incongruent sentences. We find models to opt for different strategies in understanding English and non-English homographs, highlighting a lack of a unified approach to handling cross-lingual ambiguities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09099v1">Drama Llama: An LLM-Powered Storylets Framework for Authorable Responsiveness in Interactive Narrative</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 10 pages, 5 photos
    </div>
    <details class="paper-abstract">
      In this paper, we present Drama Llama, an LLM-powered storylets framework that supports the authoring of responsive, open-ended interactive stories. DL combines the structural benefits of storylet-based systems with the generative capabilities of large language models, enabling authors to create responsive interactive narratives while maintaining narrative control. Rather than crafting complex logical preconditions in a general-purpose or domain-specific programming language, authors define triggers in natural language that fire at appropriate moments in the story. Through a preliminary authoring study with six content authors, we present initial evidence that DL can generate coherent and meaningful narratives with believable character interactions. This work suggests directions for hybrid approaches that enhance authorial control while supporting emergent narrative generation through LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09092v1">SteLLA: A Structured Grading System Using LLMs with RAG</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong general capabilities in many applications. However, how to make them reliable tools for some specific tasks such as automated short answer grading (ASAG) remains a challenge. We present SteLLA (Structured Grading System Using LLMs with RAG) in which a) Retrieval Augmented Generation (RAG) approach is used to empower LLMs specifically on the ASAG task by extracting structured information from the highly relevant and reliable external knowledge based on the instructor-provided reference answer and rubric, b) an LLM performs a structured and question-answering-based evaluation of student answers to provide analytical grades and feedback. A real-world dataset that contains students' answers in an exam was collected from a college-level Biology course. Experiments show that our proposed system can achieve substantial agreement with the human grader while providing break-down grades and feedback on all the knowledge points examined in the problem. A qualitative and error analysis of the feedback generated by GPT4 shows that GPT4 is good at capturing facts while may be prone to inferring too much implication from the given text in the grading task which provides insights into the usage of LLMs in the ASAG system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09012v1">Multimodal LLMs Can Reason about Aesthetics in Zero-Shot</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 WIP, Homepage https://github.com/songrise/MLLM4Art
    </div>
    <details class="paper-abstract">
      We present the first study on how Multimodal LLMs' (MLLMs) reasoning ability shall be elicited to evaluate the aesthetics of artworks. To facilitate this investigation, we construct MM-StyleBench, a novel high-quality dataset for benchmarking artistic stylization. We then develop a principled method for human preference modeling and perform a systematic correlation analysis between MLLMs' responses and human preference. Our experiments reveal an inherent hallucination issue of MLLMs in art evaluation, associated with response subjectivity. ArtCoT is proposed, demonstrating that art-specific task decomposition and the use of concrete language boost MLLMs' reasoning ability for aesthetics. Our findings offer valuable insights into MLLMs for art and can benefit a wide range of downstream applications, such as style transfer and artistic image generation. Code available at https://github.com/songrise/MLLM4Art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09004v1">Aegis2.0: A Diverse AI Safety Dataset and Risks Taxonomy for Alignment of LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 arXiv admin note: text overlap with arXiv:2404.05993
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) and generative AI become increasingly widespread, concerns about content safety have grown in parallel. Currently, there is a clear lack of high-quality, human-annotated datasets that address the full spectrum of LLM-related safety risks and are usable for commercial applications. To bridge this gap, we propose a comprehensive and adaptable taxonomy for categorizing safety risks, structured into 12 top-level hazard categories with an extension to 9 fine-grained subcategories. This taxonomy is designed to meet the diverse requirements of downstream users, offering more granular and flexible tools for managing various risk types. Using a hybrid data generation pipeline that combines human annotations with a multi-LLM "jury" system to assess the safety of responses, we obtain Aegis 2.0, a carefully curated collection of 34,248 samples of human-LLM interactions, annotated according to our proposed taxonomy. To validate its effectiveness, we demonstrate that several lightweight models, trained using parameter-efficient techniques on Aegis 2.0, achieve performance competitive with leading safety models fully fine-tuned on much larger, non-commercial datasets. In addition, we introduce a novel training blend that combines safety with topic following data.This approach enhances the adaptability of guard models, enabling them to generalize to new risk categories defined during inference. We plan to open-source Aegis 2.0 data and models to the research community to aid in the safety guardrailing of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05541v2">Customizable LLM-Powered Chatbot for Behavioral Science Research</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      The rapid advancement of Artificial Intelligence has resulted in the advent of Large Language Models (LLMs) with the capacity to produce text that closely resembles human communication. These models have been seamlessly integrated into diverse applications, enabling interactive and responsive communication across multiple platforms. The potential utility of chatbots transcends these traditional applications, particularly in research contexts, wherein they can offer valuable insights and facilitate the design of innovative experiments. In this study, we present a Customizable LLM-Powered Chatbot (CLPC), a web-based chatbot system designed to assist in behavioral science research. The system is meticulously designed to function as an experimental instrument rather than a conventional chatbot, necessitating users to input a username and experiment code upon access. This setup facilitates precise data cross-referencing, thereby augmenting the integrity and applicability of the data collected for research purposes. It can be easily expanded to accommodate new basic events as needed; and it allows researchers to integrate their own logging events without the necessity of implementing a separate logging mechanism. It is worth noting that our system was built to assist primarily behavioral science research but is not limited to it, it can easily be adapted to assist information retrieval research or interacting with chat bot agents in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14808v1">HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 15 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have facilitated a wide range of applications with distinct quality-of-experience requirements, from latency-sensitive online tasks, such as interactive chatbots, to throughput-focused offline tasks like document summarization. While deploying dedicated machines for these services ensures high-quality performance, it often results in resource underutilization. This paper introduces HyGen, an interference-aware LLM serving system that enables efficient co-location of online and offline workloads while preserving latency requirements. HyGen incorporates two key innovations: (1) performance control mechanisms, including a latency predictor for batch execution time estimation and an SLO-aware profiler to quantify interference, and (2) SLO-aware offline scheduling policies that maximize throughput and prevent starvation, without compromising online serving latency. Our evaluation on production workloads shows that HyGen achieves up to 5.84x higher throughput compared to existing advances while maintaining comparable latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08838v1">ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 Accepted by AAAI 2025
    </div>
    <details class="paper-abstract">
      Existing Theory of Mind (ToM) benchmarks diverge from real-world scenarios in three aspects: 1) they assess a limited range of mental states such as beliefs, 2) false beliefs are not comprehensively explored, and 3) the diverse personality traits of characters are overlooked. To address these challenges, we introduce ToMATO, a new ToM benchmark formulated as multiple-choice QA over conversations. ToMATO is generated via LLM-LLM conversations featuring information asymmetry. By employing a prompting method that requires role-playing LLMs to verbalize their thoughts before each utterance, we capture both first- and second-order mental states across five categories: belief, intention, desire, emotion, and knowledge. These verbalized thoughts serve as answers to questions designed to assess the mental states of characters within conversations. Furthermore, the information asymmetry introduced by hiding thoughts from others induces the generation of false beliefs about various mental states. Assigning distinct personality traits to LLMs further diversifies both utterances and thoughts. ToMATO consists of 5.4k questions, 753 conversations, and 15 personality trait patterns. Our analysis shows that this dataset construction approach frequently generates false beliefs due to the information asymmetry between role-playing LLMs, and effectively reflects diverse personalities. We evaluate nine LLMs on ToMATO and find that even GPT-4o mini lags behind human performance, especially in understanding false beliefs, and lacks robustness to various personality traits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03093v3">ASTER: Natural and Multi-language Unit Test Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 Accepted at ICSE-SEIP, 2025
    </div>
    <details class="paper-abstract">
      Implementing automated unit tests is an important but time-consuming activity in software development. To assist developers in this task, many techniques for automating unit test generation have been developed. However, despite this effort, usable tools exist for very few programming languages. Moreover, studies have found that automatically generated tests suffer poor readability and do not resemble developer-written tests. In this work, we present a rigorous investigation of how large language models (LLMs) can help bridge the gap. We describe a generic pipeline that incorporates static analysis to guide LLMs in generating compilable and high-coverage test cases. We illustrate how the pipeline can be applied to different programming languages, specifically Java and Python, and to complex software requiring environment mocking. We conducted an empirical study to assess the quality of the generated tests in terms of code coverage and test naturalness -- evaluating them on standard as well as enterprise Java applications and a large Python benchmark. Our results demonstrate that LLM-based test generation, when guided by static analysis, can be competitive with, and even outperform, state-of-the-art test-generation techniques in coverage achieved while also producing considerably more natural test cases that developers find easy to understand. We also present the results of a user study, conducted with 161 professional developers, that highlights the naturalness characteristics of the tests generated by our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08760v1">Leveraging LLM Agents for Translating Network Configurations</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Configuration translation is a critical and frequent task in network operations. When a network device is damaged or outdated, administrators need to replace it to maintain service continuity. The replacement devices may originate from different vendors, necessitating configuration translation to ensure seamless network operation. However, translating configurations manually is a labor-intensive and error-prone process. In this paper, we propose an intent-based framework for translating network configuration with Large Language Model (LLM) Agents. The core of our approach is an Intent-based Retrieval Augmented Generation (IRAG) module that systematically splits a configuration file into fragments, extracts intents, and generates accurate translations. We also design a two-stage verification method to validate the syntax and semantics correctness of the translated configurations. We implement and evaluate the proposed method on real-world network configurations. Experimental results show that our method achieves 97.74% syntax correctness, outperforming state-of-the-art methods in translation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08109v2">Unseen Horizons: Unveiling the Real Capability of LLM Code Generation Beyond the Familiar</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 Accepted by the 47th International Conference on Software Engineering (ICSE 2025)
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have shown strong potential in code generation tasks. However, there are still gaps before they can be fully applied in actual software development processes. Accurately assessing the code generation capabilities of large language models has become an important basis for evaluating and improving the models. Some existing works have constructed datasets to evaluate the capabilities of these models. However, the current evaluation process may encounter the illusion of "Specialist in Familiarity", primarily due to three gaps: the exposure of target code, case timeliness, and dependency availability. The fundamental reason for these gaps is that the code in current datasets may have been extensively exposed and exercised during the training phase, and due to the continuous training and development of LLM, their timeliness has been severely compromised. The key to solve the problem is to, as much as possible, evaluate the LLMs using code that they have not encountered before. Thus, the fundamental idea in this paper is to draw on the concept of code obfuscation, changing code at different levels while ensuring the functionality and output. To this end, we build a code-obfuscation based benchmark OBFUSEVAL. We first collect 1,354 raw cases from five real-world projects, including function description and code. Then we use three-level strategy (symbol, structure and semantic) to obfuscate descriptions, code and context dependencies. We evaluate four LLMs on OBFU- SEVAL and compared the effectiveness of different obfuscation strategy. We use official test suites of these projects to evaluate the generated code. The results show that after obfuscation, the average decrease ratio of test pass rate can up to 62.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08716v1">The Inherent Limits of Pretrained LLMs: The Unexpected Convergence of Instruction Tuning and In-Context Learning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 The code for this paper is available at: https://github.com/UKPLab/arxiv2025-inherent-limits-plms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), trained on extensive web-scale corpora, have demonstrated remarkable abilities across diverse tasks, especially as they are scaled up. Nevertheless, even state-of-the-art models struggle in certain cases, sometimes failing at problems solvable by young children, indicating that traditional notions of task complexity are insufficient for explaining LLM capabilities. However, exploring LLM capabilities is complicated by the fact that most widely-used models are also "instruction-tuned" to respond appropriately to prompts. With the goal of disentangling the factors influencing LLM performance, we investigate whether instruction-tuned models possess fundamentally different capabilities from base models that are prompted using in-context examples. Through extensive experiments across various model families, scales and task types, which included instruction tuning 90 different LLMs, we demonstrate that the performance of instruction-tuned models is significantly correlated with the in-context performance of their base counterparts. By clarifying what instruction-tuning contributes, we extend prior research into in-context learning, which suggests that base models use priors from pretraining data to solve tasks. Specifically, we extend this understanding to instruction-tuned models, suggesting that their pretraining data similarly sets a limiting boundary on the tasks they can solve, with the added influence of the instruction-tuning dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.15512v3">Toward Automated Simulation Research Workflow through LLM Prompt Engineering Design</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 The source code and example results of ASA can be found at https://github.com/zokaraa/autonomous_simulation_agent
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has created new opportunities for the automation of scientific research spanning both experimental processes and computational simulations. This study explores the feasibility of constructing an autonomous simulation agent (ASA) powered by LLMs through prompt engineering and automated program design to automate the entire simulation research process according to a human-provided research plan. This process includes experimental design, remote upload and simulation execution, data analysis, and report compilation. Using a well-studied simulation problem of polymer chain conformations as a test case, we assessed the long-task completion and reliability of ASAs powered by different LLMs, including GPT-4o, Claude-3.5, etc. Our findings revealed that ASA-GPT-4o achieved near-flawless execution on designated research missions, underscoring the potential of methods like ASA to achieve automation in simulation research processes to enhance research efficiency. The outlined automation can be iteratively performed for up to 20 cycles without human intervention, illustrating the potential of ASA for long-task workflow automation. Additionally, we discussed the intrinsic traits of ASA in managing extensive tasks, focusing on self-validation mechanisms, and the balance between local attention and global oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08670v1">Augmenting Smart Contract Decompiler Output through Fine-grained Dependency Analysis and LLM-facilitated Semantic Recovery</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Decompiler is a specialized type of reverse engineering tool extensively employed in program analysis tasks, particularly in program comprehension and vulnerability detection. However, current Solidity smart contract decompilers face significant limitations in reconstructing the original source code. In particular, the bottleneck of SOTA decompilers lies in inaccurate method identification, incorrect variable type recovery, and missing contract attributes. These deficiencies hinder downstream tasks and understanding of the program logic. To address these challenges, we propose SmartHalo, a new framework that enhances decompiler output by combining static analysis (SA) and large language models (LLM). SmartHalo leverages the complementary strengths of SA's accuracy in control and data flow analysis and LLM's capability in semantic prediction. More specifically, \system{} constructs a new data structure - Dependency Graph (DG), to extract semantic dependencies via static analysis. Then, it takes DG to create prompts for LLM optimization. Finally, the correctness of LLM outputs is validated through symbolic execution and formal verification. Evaluation on a dataset consisting of 465 randomly selected smart contract methods shows that SmartHalo significantly improves the quality of the decompiled code, compared to SOTA decompilers (e.g., Gigahorse). Notably, integrating GPT-4o with SmartHalo further enhances its performance, achieving precision rates of 87.39% for method boundaries, 90.39% for variable types, and 80.65% for contract attributes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16705v2">SelectIT: Selective Instruction Tuning for LLMs via Uncertainty-Aware Self-Reflection</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 Accepted to NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Instruction tuning (IT) is crucial to tailoring large language models (LLMs) towards human-centric interactions. Recent advancements have shown that the careful selection of a small, high-quality subset of IT data can significantly enhance the performance of LLMs. Despite this, common approaches often rely on additional models or data, which increases costs and limits widespread adoption. In this work, we propose a novel approach, termed SelectIT, that capitalizes on the foundational capabilities of the LLM itself. Specifically, we exploit the intrinsic uncertainty present in LLMs to more effectively select high-quality IT data, without the need for extra resources. Furthermore, we introduce a curated IT dataset, the Selective Alpaca, created by applying SelectIT to the Alpaca-GPT4 dataset. Empirical results demonstrate that IT using Selective Alpaca leads to substantial model ability enhancement. The robustness of SelectIT has also been corroborated in various foundation models and domain-specific tasks. Our findings suggest that longer and more computationally intensive IT data may serve as superior sources of IT, offering valuable insights for future research in this area. Data, code, and scripts are freely available at https://github.com/Blue-Raincoat/SelectIT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12117v3">MEMO: Fine-grained Tensor Management For Ultra-long Context LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Nowadays, Large Language Models (LLMs) have been trained using extended context lengths to foster more creative applications. However, long context training poses great challenges considering the constraint of GPU memory. It not only leads to substantial activation memory consumption during training, but also incurs considerable memory fragmentation. To facilitate long context training, existing frameworks have adopted strategies such as recomputation and various forms of parallelisms. Nevertheless, these techniques rely on redundant computation or extensive communication, resulting in low Model FLOPS Utilization (MFU). In this paper, we propose MEMO, a novel LLM training framework designed for fine-grained activation memory management. Given the quadratic scaling of computation and linear scaling of memory with sequence lengths when using FlashAttention, we offload memory-consuming activations to CPU memory after each layer's forward pass and fetch them during the backward pass. To maximize the swapping of activations without hindering computation, and to avoid exhausting limited CPU memory, we implement a token-wise activation recomputation and swapping mechanism. Furthermore, we tackle the memory fragmentation issue by employing a bi-level Mixed Integer Programming (MIP) approach, optimizing memory reuse across transformer layers. Empirical results demonstrate that MEMO achieves an average of 1.97x and 1.80x MFU compared to Megatron-LM and DeepSpeed, respectively. This improvement is attributed to MEMO's ability to minimize memory fragmentation, reduce recomputation and intensive communication, and circumvent the delays associated with the memory reorganization process due to fragmentation. By leveraging fine-grained activation memory management, MEMO facilitates efficient training of 7B LLM with 1 million sequence length on just 8 A800 GPUs, achieving an MFU of 52.30%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08631v1">SWSC: Shared Weight for Similar Channel in LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 5pages, 3 figures, work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have spurred development in multiple industries. However, the growing number of their parameters brings substantial storage and computing burdens, making it essential to explore model compression techniques for parameter reduction and easier deployment. We propose SWSC, an LLM compression method based on the concept of Shared Weight for Similar Channel. It uses the K-Means clustering algorithm to cluster model weights channel-by-channel, generating clusters with highly similar vectors within each. A representative vector from each cluster is selected to approximately replace all vectors in the cluster, significantly reducing the number of model weight parameters. However, approximate restoration will inevitably cause damage to the performance of the model. To tackle this issue, we perform singular value decomposition on the weight error values before and after compression and retain the larger singular values and their corresponding singular vectors to compensate for the accuracy. The experimental results show that our method can effectively ensure the performance of the compressed LLM even under low-precision conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08600v1">AutoRestTest: A Tool for Automated REST API Testing Using LLMs and MARL</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 To be published in the 47th IEEE/ACM International Conference on Software Engineering - Demonstration Track (ICSE-Demo 2025)
    </div>
    <details class="paper-abstract">
      As REST APIs have become widespread in modern web services, comprehensive testing of these APIs has become increasingly crucial. Due to the vast search space consisting of operations, parameters, and parameter values along with their complex dependencies and constraints, current testing tools suffer from low code coverage, leading to suboptimal fault detection. To address this limitation, we present a novel tool, AutoRestTest, which integrates the Semantic Operation Dependency Graph (SODG) with Multi-Agent Reinforcement Learning (MARL) and large language models (LLMs) for effective REST API testing. AutoRestTest determines operation-dependent parameters using the SODG and employs five specialized agents (operation, parameter, value, dependency, and header) to identify dependencies of operations and generate operation sequences, parameter combinations, and values. AutoRestTest provides a command-line interface and continuous telemetry on successful operation count, unique server errors detected, and time elapsed. Upon completion, AutoRestTest generates a detailed report highlighting errors detected and operations exercised. In this paper, we introduce our tool and present preliminary results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08579v1">What Limits LLM-based Human Simulation: LLMs or Our Design?</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      We argue that advancing LLM-based human simulation requires addressing both LLM's inherent limitations and simulation framework design challenges. Recent studies have revealed significant gaps between LLM-based human simulations and real-world observations, highlighting these dual challenges. To address these gaps, we present a comprehensive analysis of LLM limitations and our design issues, proposing targeted solutions for both aspects. Furthermore, we explore future directions that address both challenges simultaneously, particularly in data collection, LLM generation, and evaluation. To support further research in this field, we provide a curated collection of LLM-based human simulation resources.\footnote{https://github.com/Persdre/llm-human-simulation}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08558v1">LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      Teleoperating high degrees-of-freedom (DoF) robotic manipulators via low-DoF controllers like joysticks often requires frequent switching between control modes, where each mode maps controller movements to specific robot actions. Manually performing this frequent switching can make teleoperation cumbersome and inefficient. On the other hand, existing automatic mode-switching solutions, such as heuristic-based or learning-based methods, are often task-specific and lack generalizability. In this paper, we introduce LLM-Driven Automatic Mode Switching (LAMS), a novel approach that leverages Large Language Models (LLMs) to automatically switch control modes based on task context. Unlike existing methods, LAMS requires no prior task demonstrations and incrementally improves by integrating user-generated mode-switching examples. We validate LAMS through an ablation study and a user study with 10 participants on complex, long-horizon tasks, demonstrating that LAMS effectively reduces manual mode switches, is preferred over alternative methods, and improves performance over time. The project website with supplementary materials is at https://lams-assistance.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04820v2">Natural Language Outlines for Code: Literate Programming in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-01-15
    </div>
    <details class="paper-abstract">
      We propose using natural language outlines as a novel modality and interaction surface for providing AI assistance to developers throughout the software development process. An NL outline for a code function comprises multiple statements written in concise prose, which partition the code and summarize its main ideas in the style of literate programming. Crucially, we find that modern LLMs can generate accurate and high-quality NL outlines in practice. Moreover, NL outlines enable a bidirectional sync between code and NL, allowing changes in one to be automatically reflected in the other. We discuss many use cases for NL outlines: they can accelerate understanding and navigation of code and diffs, simplify code maintenance, augment code search, steer code generation, and more. We then propose and compare multiple LLM prompting techniques for generating outlines and ask professional developers to judge outline quality. Finally, we present two case studies applying NL outlines toward code review and malware detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11514v2">Counterfactual Debating with Preset Stances for Hallucination Elimination of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various natural language processing tasks but struggle with hallucination issues. Existing solutions have considered utilizing LLMs' inherent reasoning abilities to alleviate hallucination, such as self-correction and diverse sampling methods. However, these methods often overtrust LLMs' initial answers due to inherent biases. The key to alleviating this issue lies in overriding LLMs' inherent biases for answer inspection. To this end, we propose a CounterFactual Multi-Agent Debate (CFMAD) framework. CFMAD presets the stances of LLMs to override their inherent biases by compelling LLMs to generate justifications for a predetermined answer's correctness. The LLMs with different predetermined stances are engaged with a skeptical critic for counterfactual debate on the rationality of generated justifications. Finally, the debate process is evaluated by a third-party judge to determine the final answer. Extensive experiments on four datasets of three tasks demonstrate the superiority of CFMAD over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20166v2">LoL-PIM: Long-Context LLM Decoding with Scalable DRAM-PIM System</a></div>
    <div class="paper-meta">
      📅 2025-01-15
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      The expansion of large language models (LLMs) with hundreds of billions of parameters presents significant challenges to computational resources, particularly data movement and memory bandwidth. Long-context LLMs, which process sequences of tens of thousands of tokens, further increase the demand on the memory system as the complexity in attention layers and key-value cache sizes is proportional to the context length. Processing-in-Memory (PIM) maximizes memory bandwidth by moving compute to the data and can address the memory bandwidth challenges; however, PIM is not necessarily scalable to accelerate long-context LLM because of limited per-module memory capacity and the inflexibility of fixed-functional unit PIM architecture and static memory management. In this work, we propose LoL-PIM which is a multi-node PIM architecture that accelerates long context LLM through hardware-software co-design. In particular, we propose how pipeline parallelism can be exploited across a multi-PIM module while a direct PIM access (DPA) controller (or DMA for PIM) is proposed that enables dynamic PIM memory management and results in efficient PIM utilization across a diverse range of context length. We developed an MLIR-based compiler for LoL-PIM extending a commercial PIM-based compiler where the software modifications were implemented and evaluated, while the hardware changes were modeled in the simulator. Our evaluations demonstrate that LoL-PIM significantly improves throughput and reduces latency for long-context LLM inference, outperforming both multi-GPU and GPU-PIM systems (up to 8.54x and 16.0x speedup, respectively), thereby enabling more efficient deployment of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03205v3">U-MATH: A University-Level Benchmark for Evaluating Mathematical Skills in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      The current evaluation of mathematical skills in LLMs is limited, as existing benchmarks are either relatively small, primarily focus on elementary and high-school problems, or lack diversity in topics. Additionally, the inclusion of visual elements in tasks remains largely under-explored. To address these gaps, we introduce U-MATH, a novel benchmark of 1,100 unpublished open-ended university-level problems sourced from teaching materials. It is balanced across six core subjects, with 20% of multimodal problems. Given the open-ended nature of U-MATH problems, we employ an LLM to judge the correctness of generated solutions. To this end, we release $\mu$-MATH, a dataset to evaluate the LLMs' capabilities in judging solutions. The evaluation of general domain, math-specific, and multimodal LLMs highlights the challenges presented by U-MATH. Our findings reveal that LLMs achieve a maximum accuracy of only 63% on text-based tasks, with even lower 45% on visual problems. The solution assessment proves challenging for LLMs, with the best LLM judge having an F1-score of 80% on $\mu$-MATH.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08322v1">Exploring Robustness of Multilingual LLMs on Real-World Noisy Data</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are trained on Web data that might contain spelling errors made by humans. But do they become robust to similar real-world noise? In this paper, we investigate the effect of real-world spelling mistakes on the performance of 9 language models, with parameters ranging from 0.2B to 13B, in 3 different NLP tasks, namely Natural Language Inference (NLI), Name Entity Recognition (NER), and Intent Classification (IC). We perform our experiments on 6 different languages and build a dictionary of real-world noise for them using the Wikipedia edit history. We show that the performance gap of the studied models on the clean and noisy test data averaged across all the datasets and languages ranges from 2.3 to 4.3 absolute percentage points. In addition, mT5 models, in general, show more robustness compared to BLOOM, Falcon, and BERT-like models. In particular, mT5 (13B), was the most robust on average overall, across the 3 tasks, and in 4 of the 6 languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08292v1">HALoGEN: Fantastic LLM Hallucinations and Where to Find Them</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Despite their impressive ability to generate high-quality and fluent text, generative large language models (LLMs) also produce hallucinations: statements that are misaligned with established world knowledge or provided input context. However, measuring hallucination can be challenging, as having humans verify model generations on-the-fly is both expensive and time-consuming. In this work, we release HALoGEN, a comprehensive hallucination benchmark consisting of: (1) 10,923 prompts for generative models spanning nine domains including programming, scientific attribution, and summarization, and (2) automatic high-precision verifiers for each use case that decompose LLM generations into atomic units, and verify each unit against a high-quality knowledge source. We use this framework to evaluate ~150,000 generations from 14 language models, finding that even the best-performing models are riddled with hallucinations (sometimes up to 86% of generated atomic facts depending on the domain). We further define a novel error classification for LLM hallucinations based on whether they likely stem from incorrect recollection of training data (Type A errors), or incorrect knowledge in training data (Type B errors), or are fabrication (Type C errors). We hope our framework provides a foundation to enable the principled study of why generative models hallucinate, and advances the development of trustworthy large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08276v1">Exploring Robustness of LLMs to Sociodemographically-Conditioned Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance in various NLP tasks. However, there are concerns about their reliability in different domains of linguistic variations. Many works have proposed robustness evaluation measures for local adversarial attacks, but we need globally robust models unbiased to different language styles. We take a broader approach to explore a wider range of variations across sociodemographic dimensions to perform structured reliability tests on the reasoning capacity of language models. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic styles. The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their reasoning capabilities in real-world, complex language scenarios. We also explore measures such as perplexity, explainability, and ATOMIC performance of paraphrases for fine-grained reliability analysis of LLMs on these sets. We find that demographic-specific paraphrasing significantly impacts the performance of language models, indicating that the subtleties of language variations remain a significant challenge. The code and dataset will be made available for reproducibility and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08262v1">Addressing the sustainable AI trilemma: a case study on LLM agents and RAG</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities, but their widespread deployment and more advanced applications raise critical sustainability challenges, particularly in inference energy consumption. We propose the concept of the Sustainable AI Trilemma, highlighting the tensions between AI capability, digital equity, and environmental sustainability. Through a systematic case study of LLM agents and retrieval-augmented generation (RAG), we analyze the energy costs embedded in memory module designs and introduce novel metrics to quantify the trade-offs between energy consumption and system performance. Our experimental results reveal significant energy inefficiencies in current memory-augmented frameworks and demonstrate that resource-constrained environments face disproportionate efficiency penalties. Our findings challenge the prevailing LLM-centric paradigm in agent design and provide practical insights for developing more sustainable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08365v1">Towards Best Practices for Open Datasets for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Many AI companies are training their large language models (LLMs) on data without the permission of the copyright owners. The permissibility of doing so varies by jurisdiction: in countries like the EU and Japan, this is allowed under certain restrictions, while in the United States, the legal landscape is more ambiguous. Regardless of the legal status, concerns from creative producers have led to several high-profile copyright lawsuits, and the threat of litigation is commonly cited as a reason for the recent trend towards minimizing the information shared about training datasets by both corporate and public interest actors. This trend in limiting data information causes harm by hindering transparency, accountability, and innovation in the broader ecosystem by denying researchers, auditors, and impacted individuals access to the information needed to understand AI models. While this could be mitigated by training language models on open access and public domain data, at the time of writing, there are no such models (trained at a meaningful scale) due to the substantial technical and sociological challenges in assembling the necessary corpus. These challenges include incomplete and unreliable metadata, the cost and complexity of digitizing physical records, and the diverse set of legal and technical skills required to ensure relevance and responsibility in a quickly changing landscape. Building towards a future where AI systems can be trained on openly licensed data that is responsibly curated and governed requires collaboration across legal, technical, and policy domains, along with investments in metadata standards, digitization, and fostering a culture of openness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08219v1">Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown significant improvements in many natural language processing (NLP) tasks, accelerating their rapid adoption across many industries. These models are resource-intensive, requiring extensive computational resources both during training and inference, leading to increased energy consumption and negative environmental impact. As their adoption accelerates, the sustainability of LLMs has become a critical issue, necessitating strategies to optimize their runtime efficiency without compromising performance. Hence, it is imperative to identify the parameters that significantly influence the performance and energy efficiency of LLMs. To that end, in this work, we investigate the effect of important parameters on the performance and energy efficiency of LLMs during inference and examine their trade-offs. First, we analyze how different types of models with varying numbers of parameters and architectures perform on tasks like text generation, question answering, and summarization by benchmarking LLMs such as Falcon-7B, Mistral-7B-v0.1, T5-3B, GPT-2, GPT-J-6B, and GPT-Neo-2.7B. Second, we study input and output sequence characteristics such as sequence length concerning energy consumption, performance, and throughput. Finally, we explore the impact of hardware-based power-saving techniques, i.e., Dynamic Voltage Frequency Scaling (DVFS), on the models' latency and energy efficiency. Our extensive benchmarking and statistical analysis reveal many interesting findings, uncovering how specific optimizations can reduce energy consumption while maintaining throughput and accuracy. This study provides actionable insights for researchers and practitioners to design energy-efficient LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08203v1">ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. In this work, we propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of seven LLMs, including LLama3, Mistral, and Mathstral, on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03565v3">Personalized LLM Response Generation with Parameterized Memory Injection</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable proficiency in comprehending and generating natural language. On the other hand, personalized LLM response generation holds the potential to offer substantial benefits for individuals in critical areas such as medical. Existing research has explored memory-augmented methods to prompt the LLM with pre-stored user-specific knowledge for personalized response generation in terms of new queries. We contend that such paradigm is unable to perceive fine-granularity information. In this study, we propose a novel \textbf{M}emory-\textbf{i}njected approach using parameter-efficient fine-tuning (PEFT) and along with a Bayesian Optimisation searching strategy to achieve \textbf{L}LM \textbf{P}ersonalization(\textbf{MiLP}).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08200v1">CWEval: Outcome-driven Evaluation on Functionality and Security of LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 to be published in LLM4Code 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly aided developers by generating or assisting in code writing, enhancing productivity across various tasks. While identifying incorrect code is often straightforward, detecting vulnerabilities in functionally correct code is more challenging, especially for developers with limited security knowledge, which poses considerable security risks of using LLM-generated code and underscores the need for robust evaluation benchmarks that assess both functional correctness and security. Current benchmarks like CyberSecEval and SecurityEval attempt to solve it but are hindered by unclear and impractical specifications, failing to assess both functionality and security accurately. To tackle these deficiencies, we introduce CWEval, a novel outcome-driven evaluation framework designed to enhance the evaluation of secure code generation by LLMs. This framework not only assesses code functionality but also its security simultaneously with high-quality task specifications and outcome-driven test oracles which provides high accuracy. Coupled with CWEval-bench, a multilingual, security-critical coding benchmark, CWEval provides a rigorous empirical security evaluation on LLM-generated code, overcoming previous benchmarks' shortcomings. Through our evaluations, CWEval reveals a notable portion of functional but insecure code produced by LLMs, and shows a serious inaccuracy of previous evaluations, ultimately contributing significantly to the field of secure code generation. We open-source our artifact at: https://github.com/Co1lin/CWEval .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08197v1">OpenCSG Chinese Corpus: A Series of High-quality Chinese Datasets for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 The datasets are available on https://huggingface.co/collections/opencsg/chinese-fineweb-66cfed105f502ece8f29643e ; The code is on https://github.com/yuyijiong/fineweb-edu-chinese
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities, but their success heavily relies on the quality of pretraining corpora. For Chinese LLMs, the scarcity of high-quality Chinese datasets presents a significant challenge, often limiting their performance. To address this issue, we propose the OpenCSG Chinese Corpus, a series of high-quality datasets specifically designed for LLM pretraining, post-training, and fine-tuning. This corpus includes Fineweb-edu-chinese, Fineweb-edu-chinese-v2, Cosmopedia-chinese, and Smoltalk-chinese, each with distinct characteristics: Fineweb-edu datasets focus on filtered, high-quality content derived from diverse Chinese web sources; Cosmopedia-chinese provides synthetic, textbook-style data for knowledge-intensive training; and Smoltalk-chinese emphasizes stylistic and diverse chat-format data. The OpenCSG Chinese Corpus is characterized by its high-quality text, diverse coverage across domains, and scalable, reproducible data curation processes. Additionally, we conducted extensive experimental analyses, including evaluations on smaller parameter models, which demonstrated significant performance improvements in tasks such as C-Eval, showcasing the effectiveness of the corpus for training Chinese LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08192v1">PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used across various applications, but their substantial computational requirements pose significant challenges, particularly in terms of HBM bandwidth bottlenecks and inter-device communication overhead. In this paper, we present PRESERVE, a novel prefetching framework designed to optimize LLM inference by overlapping memory reads for model weights and KV-cache with collective communication operations. Through extensive experiments conducted on commercial AI accelerators, we demonstrate up to 1.6x end-to-end speedup on state-of-the-art, open-source LLMs. Additionally, we perform a design space exploration that identifies the optimal hardware configuration for the proposed method, showing a further 1.25x improvement in performance per cost by selecting the optimal L2 cache size. Our results show that PRESERVE has the potential to mitigate the memory bottlenecks and communication overheads, offering a solution to improve the performance and scalability of the LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07572v2">WebWalker: Benchmarking LLMs in Web Traversal</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) demonstrates remarkable performance across tasks in open-domain question-answering. However, traditional search engines may retrieve shallow content, limiting the ability of LLMs to handle complex, multi-layered information. To address it, we introduce WebWalkerQA, a benchmark designed to assess the ability of LLMs to perform web traversal. It evaluates the capacity of LLMs to traverse a website's subpages to extract high-quality data systematically. We propose WebWalker, which is a multi-agent framework that mimics human-like web navigation through an explore-critic paradigm. Extensive experimental results show that WebWalkerQA is challenging and demonstrates the effectiveness of RAG combined with WebWalker, through the horizontal and vertical integration in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16779v2">Inductive Learning of Logical Theories with LLMs: An Expressivity-Graded Analysis</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      This work presents a novel systematic methodology to analyse the capabilities and limitations of Large Language Models (LLMs) with feedback from a formal inference engine, on logic theory induction. The analysis is complexity-graded w.r.t. rule dependency structure, allowing quantification of specific inference challenges on LLM performance. Integrating LLMs with formal methods is a promising frontier in the Natural Language Processing field, as an important avenue for improving model inference control and explainability. In particular, inductive learning over complex sets of facts and rules, poses unique challenges for current autoregressive models, as they lack explicit symbolic grounding. While they can be complemented by formal systems, the properties delivered by LLMs regarding inductive learning, are not well understood and quantified. Empirical results indicate that the largest LLMs can achieve competitive results against a SOTA Inductive Logic Programming (ILP) system baseline, but also that tracking long predicate relationship chains is a more difficult obstacle than theory complexity for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13612v2">Are LLMs Good Literature Review Writers? Evaluating the Literature Review Writing Ability of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 12 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The literature review is a crucial form of academic writing that involves complex processes of literature collection, organization, and summarization. The emergence of large language models (LLMs) has introduced promising tools to automate these processes. However, their actual capabilities in writing comprehensive literature reviews remain underexplored, such as whether they can generate accurate and reliable references. To address this gap, we propose a framework to assess the literature review writing ability of LLMs automatically. We evaluate the performance of LLMs across three tasks: generating references, writing abstracts, and writing literature reviews. We employ external tools for a multidimensional evaluation, which includes assessing hallucination rates in references, semantic coverage, and factual consistency with human-written context. By analyzing the experimental results, we find that, despite advancements, even the most sophisticated models still cannot avoid generating hallucinated references. Additionally, different models exhibit varying performance in literature review writing across different disciplines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08068v1">A Roadmap to Guide the Integration of LLMs in Hierarchical Planning</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 5 pages, 0 figures, to be published in the AAAI Workshop on Planning in the Era of LLMs ( https://llmforplanning.github.io )
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) are fostering their integration into several reasoning-related fields, including Automated Planning (AP). However, their integration into Hierarchical Planning (HP), a subfield of AP that leverages hierarchical knowledge to enhance planning performance, remains largely unexplored. In this preliminary work, we propose a roadmap to address this gap and harness the potential of LLMs for HP. To this end, we present a taxonomy of integration methods, exploring how LLMs can be utilized within the HP life cycle. Additionally, we provide a benchmark with a standardized dataset for evaluating the performance of future LLM-based HP approaches, and present initial results for a state-of-the-art HP planner and LLM planner. As expected, the latter exhibits limited performance (3\% correct plans, and none with a correct hierarchical decomposition) but serves as a valuable baseline for future approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03335v2">Audio-Agent: Leveraging LLMs For Audio Generation, Editing and Composition</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      We introduce Audio-Agent, a multimodal framework for audio generation, editing and composition based on text or video inputs. Conventional approaches for text-to-audio (TTA) tasks often make single-pass inferences from text descriptions. While straightforward, this design struggles to produce high-quality audio when given complex text conditions. In our method, we utilize a pre-trained TTA diffusion network as the audio generation agent to work in tandem with GPT-4, which decomposes the text condition into atomic, specific instructions and calls the agent for audio generation. In doing so, Audio-Agent can generate high-quality audio that is closely aligned with the provided text or video exhibiting complex and multiple events, while supporting variable-length and variable-volume generation. For video-to-audio (VTA) tasks, most existing methods require training a timestamp detector to synchronize video events with the generated audio, a process that can be tedious and time-consuming. Instead, we propose a simpler approach by fine-tuning a pre-trained Large Language Model (LLM), e.g., Gemma2-2B-it, to obtain both semantic and temporal conditions that bridge the video and audio modality. Consequently, our framework contributes a comprehensive solution for both TTA and VTA tasks without substantial computational overhead in training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07927v1">Gandalf the Red: Adaptive Security for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 Niklas Pfister, V\'aclav Volhejn and Manuel Knott contributed equally
    </div>
    <details class="paper-abstract">
      Current evaluations of defenses against prompt attacks in large language model (LLM) applications often overlook two critical factors: the dynamic nature of adversarial behavior and the usability penalties imposed on legitimate users by restrictive defenses. We propose D-SEC (Dynamic Security Utility Threat Model), which explicitly separates attackers from legitimate users, models multi-step interactions, and rigorously expresses the security-utility in an optimizable form. We further address the shortcomings in existing evaluations by introducing Gandalf, a crowd-sourced, gamified red-teaming platform designed to generate realistic, adaptive attack datasets. Using Gandalf, we collect and release a dataset of 279k prompt attacks. Complemented by benign user data, our analysis reveals the interplay between security and utility, showing that defenses integrated in the LLM (e.g., system prompts) can degrade usability even without blocking requests. We demonstrate that restricted application domains, defense-in-depth, and adaptive defenses are effective strategies for building secure and useful LLM applications. Code is available at \href{https://github.com/lakeraai/dsec-gandalf}{\texttt{https://github.com/lakeraai/dsec-gandalf}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07892v1">Leveraging Metamemory Mechanisms for Enhanced Data-Free Code Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 11 pages,6 figures
    </div>
    <details class="paper-abstract">
      Automated code generation using large language models (LLMs) has gained attention due to its efficiency and adaptability. However, real-world coding tasks or benchmarks like HumanEval and StudentEval often lack dedicated training datasets, challenging existing few-shot prompting approaches that rely on reference examples. Inspired by human metamemory-a cognitive process involving recall and evaluation-we present a novel framework (namely M^2WF) for improving LLMs' one-time code generation. This approach enables LLMs to autonomously generate, evaluate, and utilize synthetic examples to enhance reliability and performance. Unlike prior methods, it minimizes dependency on curated data and adapts flexibly to various coding scenarios. Our experiments demonstrate significant improvements in coding benchmarks, offering a scalable and robust solution for data-free environments. The code and framework will be publicly available on GitHub and HuggingFace.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.03852v3">FLM-101B: An Open LLM and How to Train It with $100K Budget</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are considered important approaches towards foundational machine intelligence, achieving remarkable success in Natural Language Processing and multimodal tasks, among others. However, the carbon footprints and financial costs originating from heavy pre-training computation is a non-negligible issue. Progressive training methods, inspired by the neurogenesis process that grows neural structures, have shown potential to accelerate LLM pre-training. However, the algorithms, implementation, and practices for progressively training LLMs beyond 100B parameters remain underexplored. In this paper, we show that our model, namely FLM-101B, trained with our growth strategy under a budget of \$100K, reaches 80\% of the baselines' performances with only 10\% of their floating-point operations. We believe that further studies on progressive training will benefit the community by cutting down the costs and promoting green AI. The checkpoint of FLM-101B is released at https://huggingface.co/CofeAI/FLM-101B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09012v2">What Makes Cryptic Crosswords Challenging for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 COLING 2025. arXiv admin note: text overlap with arXiv:2403.12094
    </div>
    <details class="paper-abstract">
      Cryptic crosswords are puzzles that rely on general knowledge and the solver's ability to manipulate language on different levels, dealing with various types of wordplay. Previous research suggests that solving such puzzles is challenging even for modern NLP models, including Large Language Models (LLMs). However, there is little to no research on the reasons for their poor performance on this task. In this paper, we establish the benchmark results for three popular LLMs: Gemma2, LLaMA3 and ChatGPT, showing that their performance on this task is still significantly below that of humans. We also investigate why these models struggle to achieve superior performance. We release our code and introduced datasets at https://github.com/bodasadallah/decrypting-crosswords.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07857v1">Hierarchical Repository-Level Code Summarization for Business Applications Using Local LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 To appear at LLM4Code@ICSE 2025
    </div>
    <details class="paper-abstract">
      In large-scale software development, understanding the functionality and intent behind complex codebases is critical for effective development and maintenance. While code summarization has been widely studied, existing methods primarily focus on smaller code units, such as functions, and struggle with larger code artifacts like files and packages. Additionally, current summarization models tend to emphasize low-level implementation details, often overlooking the domain and business context that are crucial for real-world applications. This paper proposes a two-step hierarchical approach for repository-level code summarization, tailored to business applications. First, smaller code units such as functions and variables are identified using syntax analysis and summarized with local LLMs. These summaries are then aggregated to generate higher-level file and package summaries. To ensure the summaries are grounded in business context, we design custom prompts that capture the intended purpose of code artifacts based on the domain and problem context of the business application. We evaluate our approach on a business support system (BSS) for the telecommunications domain, showing that syntax analysis-based hierarchical summarization improves coverage, while business-context grounding enhances the relevance of the generated summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07845v1">Reasoning with Graphs: Structuring Implicit Knowledge to Enhance LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across a wide range of tasks; however, they still encounter challenges in reasoning tasks that require understanding and inferring relationships between distinct pieces of information within text sequences. This challenge is particularly pronounced in tasks involving multi-step processes, such as logical reasoning and multi-hop question answering, where understanding implicit relationships between entities and leveraging multi-hop connections in the given context are crucial. Graphs, as fundamental data structures, explicitly represent pairwise relationships between entities, thereby offering the potential to enhance LLMs' reasoning capabilities. External graphs have proven effective in supporting LLMs across multiple tasks. However, in many reasoning tasks, no pre-existing graph structure is provided. Can we structure implicit knowledge derived from context into graphs to assist LLMs in reasoning? In this paper, we propose Reasoning with Graphs (RwG) by first constructing explicit graphs from the context and then leveraging these graphs to enhance LLM reasoning performance on reasoning tasks. Extensive experiments demonstrate the effectiveness of the proposed method in improving both logical reasoning and multi-hop question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07811v1">CodeCoR: An LLM-Based Self-Reflective Multi-Agent Framework for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-14
    </div>
    <details class="paper-abstract">
      Code generation aims to produce code that fulfills requirements written in natural languages automatically. Large language Models (LLMs) like ChatGPT have demonstrated promising effectiveness in this area. Nonetheless, these LLMs often fail to ensure the syntactic and semantic correctness of the generated code. Recently, researchers proposed multi-agent frameworks that guide LLMs with different prompts to analyze programming tasks, generate code, perform testing in a sequential workflow. However, the performance of the workflow is not robust as the code generation depends on the performance of each agent. To address this challenge, we propose CodeCoR, a self-reflective multi-agent framework that evaluates the effectiveness of each agent and their collaborations. Specifically, for a given task description, four agents in CodeCoR generate prompts, code, test cases, and repair advice, respectively. Each agent generates more than one output and prunes away the low-quality ones. The generated code is tested in the local environment: the code that fails to pass the generated test cases is sent to the repair agent and the coding agent re-generates the code based on repair advice. Finally, the code that passes the most number of generated test cases is returned to users. Our experiments on four widely used datasets, HumanEval, HumanEval-ET, MBPP, and MBPP-ET, demonstrate that CodeCoR significantly outperforms existing baselines (e.g., CodeCoT and MapCoder), achieving an average Pass@1 score of 77.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09008v2">LLM Reading Tea Leaves: Automatically Evaluating Topic Models with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-01-14
      | 💬 Forthcoming in Transactions of the Association for Computational Linguistics (TACL) published by MIT Press
    </div>
    <details class="paper-abstract">
      Topic modeling has been a widely used tool for unsupervised text analysis. However, comprehensive evaluations of a topic model remain challenging. Existing evaluation methods are either less comparable across different models (e.g., perplexity) or focus on only one specific aspect of a model (e.g., topic quality or document representation quality) at a time, which is insufficient to reflect the overall model performance. In this paper, we propose WALM (Word Agreement with Language Model), a new evaluation method for topic modeling that considers the semantic quality of document representations and topics in a joint manner, leveraging the power of Large Language Models (LLMs). With extensive experiments involving different types of topic models, WALM is shown to align with human judgment and can serve as a complementary evaluation method to the existing ones, bringing a new perspective to topic modeling. Our software package is available at https://github.com/Xiaohao-Yang/Topic_Model_Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14634v3">Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Added supplementary material
    </div>
    <details class="paper-abstract">
      The scientific ideation process often involves blending salient aspects of existing papers to create new ideas. To see if large language models (LLMs) can assist this process, we contribute Scideator, a novel mixed-initiative tool for scientific ideation. Starting from a user-provided set of papers, Scideator extracts key facets (purposes, mechanisms, and evaluations) from these and relevant papers, allowing users to explore the idea space by interactively recombining facets to synthesize inventive ideas. Scideator also helps users to gauge idea novelty by searching the literature for potential overlaps and showing automated novelty assessments and explanations. To support these tasks, Scideator introduces four LLM-powered retrieval-augmented generation (RAG) modules: Analogous Paper Facet Finder, Faceted Idea Generator, Idea Novelty Checker, and Idea Novelty Iterator. In a within-subjects user study, 19 computer-science researchers identified significantly more interesting ideas using Scideator compared to a strong baseline combining a scientific search engine with LLM interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07663v1">Enhancing Talent Employment Insights Through Feature Extraction with LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      This paper explores the application of large language models (LLMs) to extract nuanced and complex job features from unstructured job postings. Using a dataset of 1.2 million job postings provided by AdeptID, we developed a robust pipeline to identify and classify variables such as remote work availability, remuneration structures, educational requirements, and work experience preferences. Our methodology combines semantic chunking, retrieval-augmented generation (RAG), and fine-tuning DistilBERT models to overcome the limitations of traditional parsing tools. By leveraging these techniques, we achieved significant improvements in identifying variables often mislabeled or overlooked, such as non-salary-based compensation and inferred remote work categories. We present a comprehensive evaluation of our fine-tuned models and analyze their strengths, limitations, and potential for scaling. This work highlights the promise of LLMs in labor market analytics, providing a foundation for more accurate and actionable insights into job data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07425v1">Enhancing LLM's Ability to Generate More Repository-Aware Unit Tests Through Precise Contextual Information Injection</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Though many learning-based approaches have been proposed for unit test generation and achieved remarkable performance, they still have limitations in relying on task-specific datasets. Recently, Large Language Models (LLMs) guided by prompt engineering have gained attention for their ability to handle a broad range of tasks, including unit test generation. Despite their success, LLMs may exhibit hallucinations when generating unit tests for focal methods or functions due to their lack of awareness regarding the project's global context. These hallucinations may manifest as calls to non-existent methods, as well as incorrect parameters or return values, such as mismatched parameter types or numbers. While many studies have explored the role of context, they often extract fixed patterns of context for different models and focal methods, which may not be suitable for all generation processes (e.g., excessive irrelevant context could lead to redundancy, preventing the model from focusing on essential information). To overcome this limitation, we propose RATester, which enhances the LLM's ability to generate more repository-aware unit tests through global contextual information injection. To equip LLMs with global knowledge similar to that of human testers, we integrate the language server gopls, which provides essential features (e.g., definition lookup) to assist the LLM. When RATester encounters an unfamiliar identifier (e.g., an unfamiliar struct name), it first leverages gopls to fetch relevant definitions and documentation comments, and then uses this global knowledge to guide the LLM. By utilizing gopls, RATester enriches the LLM's knowledge of the project's global context, thereby reducing hallucinations during unit test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11547v2">Small Language Models can Outperform Humans in Short Creative Writing: A Study Comparing SLMs with Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Accepted as Main Conference Paper at COLING 2025
    </div>
    <details class="paper-abstract">
      In this paper, we evaluate the creative fiction writing abilities of a fine-tuned small language model (SLM), BART-large, and compare its performance to human writers and two large language models (LLMs): GPT-3.5 and GPT-4o. Our evaluation consists of two experiments: (i) a human study in which 68 participants rated short stories from humans and the SLM on grammaticality, relevance, creativity, and attractiveness, and (ii) a qualitative linguistic analysis examining the textual characteristics of stories produced by each model. In the first experiment, BART-large outscored average human writers overall (2.11 vs. 1.85), a 14% relative improvement, though the slight human advantage in creativity was not statistically significant. In the second experiment, qualitative analysis showed that while GPT-4o demonstrated near-perfect coherence and used less cliche phrases, it tended to produce more predictable language, with only 3% of its synopses featuring surprising associations (compared to 15% for BART). These findings highlight how model size and fine-tuning influence the balance between creativity, fluency, and coherence in creative writing tasks, and demonstrate that smaller models can, in certain contexts, rival both humans and larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09536v2">Galapagos: Automated N-Version Programming with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      N-Version Programming is a well-known methodology for developing fault-tolerant systems. It achieves fault detection and correction at runtime by adding diverse redundancy into programs, minimizing fault mode overlap between redundant program variants. In this work, we propose the automated generation of program variants using large language models. We design, develop and evaluate Gal\'apagos: a tool for generating program variants using LLMs, validating their correctness and equivalence, and using them to assemble N-Version binaries. We evaluate Gal\'apagos by creating N-Version components of real-world C code. Our original results show that Gal\'apagos can produce program variants that are proven to be functionally equivalent, even when the variants are written in a different programming language. Our systematic diversity measurement indicates that functionally equivalent variants produced by Gal\'apagos, are statically different after compilation, and present diverging internal behavior at runtime. We demonstrate that the variants produced by Gal\'apagos can protect C code against real miscompilation bugs which affect the Clang compiler. Overall, our paper shows that producing N-Version software can be drastically automated by advanced usage of practical formal verification and generative language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07314v1">FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 11 pages, 4 figures, 4 tables. To be published in NoDaLiDa/Baltic-HLT 2025 proceedings
    </div>
    <details class="paper-abstract">
      Data quality is crucial for training Large Language Models (LLMs). Traditional heuristic filters often miss low-quality text or mistakenly remove valuable content. In this paper, we introduce an LLM-based line-level filtering method to enhance training data quality. We use GPT-4o mini to label a 20,000-document sample from FineWeb at the line level, allowing the model to create descriptive labels for low-quality lines. These labels are grouped into nine main categories, and we train a DeBERTa-v3 classifier to scale the filtering to a 10B-token subset of FineWeb. To test the impact of our filtering, we train GPT-2 models on both the original and the filtered datasets. The results show that models trained on the filtered data achieve higher accuracy on the HellaSwag benchmark and reach their performance targets faster, even with up to 25\% less data. This demonstrates that LLM-based line-level filtering can significantly improve data quality and training efficiency for LLMs. We release our quality-annotated dataset, FinerWeb-10BT, and the codebase to support further work in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07288v1">LLM-Net: Democratizing LLMs-as-a-Service through Blockchain-based Expert Networks</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      The centralization of Large Language Models (LLMs) development has created significant barriers to AI advancement, limiting the democratization of these powerful technologies. This centralization, coupled with the scarcity of high-quality training data and mounting complexity of maintaining comprehensive expertise across rapidly expanding knowledge domains, poses critical challenges to the continued growth of LLMs. While solutions like Retrieval-Augmented Generation (RAG) offer potential remedies, maintaining up-to-date expert knowledge across diverse domains remains a significant challenge, particularly given the exponential growth of specialized information. This paper introduces LLMs Networks (LLM-Net), a blockchain-based framework that democratizes LLMs-as-a-Service through a decentralized network of specialized LLM providers. By leveraging collective computational resources and distributed domain expertise, LLM-Net incorporates fine-tuned expert models for various specific domains, ensuring sustained knowledge growth while maintaining service quality through collaborative prompting mechanisms. The framework's robust design includes blockchain technology for transparent transaction and performance validation, establishing an immutable record of service delivery. Our simulation, built on top of state-of-the-art LLMs such as Claude 3.5 Sonnet, Llama 3.1, Grok-2, and GPT-4o, validates the effectiveness of the reputation-based mechanism in maintaining service quality by selecting high-performing respondents (LLM providers). Thereby it demonstrates the potential of LLM-Net to sustain AI advancement through the integration of decentralized expertise and blockchain-based accountability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07267v1">Transforming Role Classification in Scientific Teams Using LLMs and Advanced Predictive Analytics</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 14 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Scientific team dynamics are critical in determining the nature and impact of research outputs. However, existing methods for classifying author roles based on self-reports and clustering lack comprehensive contextual analysis of contributions. Thus, we present a transformative approach to classifying author roles in scientific teams using advanced large language models (LLMs), which offers a more refined analysis compared to traditional clustering methods. Specifically, we seek to complement and enhance these traditional methods by utilizing open source and proprietary LLMs, such as GPT-4, Llama3 70B, Llama2 70B, and Mistral 7x8B, for role classification. Utilizing few-shot prompting, we categorize author roles and demonstrate that GPT-4 outperforms other models across multiple categories, surpassing traditional approaches such as XGBoost and BERT. Our methodology also includes building a predictive deep learning model using 10 features. By training this model on a dataset derived from the OpenAlex database, which provides detailed metadata on academic publications -- such as author-publication history, author affiliation, research topics, and citation counts -- we achieve an F1 score of 0.76, demonstrating robust classification of author roles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.12094v2">Are LLMs Good Cryptic Crossword Solvers?</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Cryptic crosswords are puzzles that rely not only on general knowledge but also on the solver's ability to manipulate language on different levels and deal with various types of wordplay. Previous research suggests that solving such puzzles is a challenge even for modern NLP models. However, the abilities of large language models (LLMs) have not yet been tested on this task. In this paper, we establish the benchmark results for three popular LLMs -- LLaMA2, Mistral, and ChatGPT -- showing that their performance on this task is still far from that of humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07237v1">Breaking Memory Limits: Gradient Wavelet Transform Enhances LLMs Training</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive performance across a range of natural language processing tasks. However, their vast number of parameters introduces significant memory challenges during training, particularly when using memory-intensive optimizers like Adam. Existing memory-efficient algorithms often rely on techniques such as singular value decomposition projection or weight freezing. While these approaches help alleviate memory constraints, they generally produce suboptimal results compared to full-rank updates. In this paper, we investigate the memory-efficient method beyond low-rank training, proposing a novel solution called Gradient Wavelet Transform (GWT), which applies wavelet transforms to gradients in order to significantly reduce the memory requirements for maintaining optimizer states. We demonstrate that GWT can be seamlessly integrated with memory-intensive optimizers, enabling efficient training without sacrificing performance. Through extensive experiments on both pre-training and fine-tuning tasks, we show that GWT achieves state-of-the-art performance compared with advanced memory-efficient optimizers and full-rank approaches in terms of both memory usage and training performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07224v1">Touched by ChatGPT: Using an LLM to Drive Affective Tactile Interaction</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Touch is a fundamental aspect of emotion-rich communication, playing a vital role in human interaction and offering significant potential in human-robot interaction. Previous research has demonstrated that a sparse representation of human touch can effectively convey social tactile signals. However, advances in human-robot tactile interaction remain limited, as many humanoid robots possess simplistic capabilities, such as only opening and closing their hands, restricting nuanced tactile expressions. In this study, we explore how a robot can use sparse representations of tactile vibrations to convey emotions to a person. To achieve this, we developed a wearable sleeve integrated with a 5x5 grid of vibration motors, enabling the robot to communicate diverse tactile emotions and gestures. Using chain prompts within a Large Language Model (LLM), we generated distinct 10-second vibration patterns corresponding to 10 emotions (e.g., happiness, sadness, fear) and 6 touch gestures (e.g., pat, rub, tap). Participants (N = 32) then rated each vibration stimulus based on perceived valence and arousal. People are accurate at recognising intended emotions, a result which aligns with earlier findings. These results highlight the LLM's ability to generate emotional haptic data and effectively convey emotions through tactile signals. By translating complex emotional and tactile expressions into vibratory patterns, this research demonstrates how LLMs can enhance physical interaction between humans and robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18873v3">LayoutCopilot: An LLM-powered Multi-agent Collaborative Framework for Interactive Analog Layout Design</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 8pages, 8figures
    </div>
    <details class="paper-abstract">
      Analog layout design heavily involves interactive processes between humans and design tools. Electronic Design Automation (EDA) tools for this task are usually designed to use scripting commands or visualized buttons for manipulation, especially for interactive automation functionalities, which have a steep learning curve and cumbersome user experience, making a notable barrier to designers' adoption. Aiming to address such a usability issue, this paper introduces LayoutCopilot, a pioneering multi-agent collaborative framework powered by Large Language Models (LLMs) for interactive analog layout design. LayoutCopilot simplifies human-tool interaction by converting natural language instructions into executable script commands, and it interprets high-level design intents into actionable suggestions, significantly streamlining the design process. Experimental results demonstrate the flexibility, efficiency, and accessibility of LayoutCopilot in handling real-world analog designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.14047v3">An empirical study of LLaMA3 quantization: from LLMs to MLLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      The LLaMA family, a collection of foundation language models ranging from 7B to 65B parameters, has become one of the most powerful open-source large language models (LLMs) and the popular LLM backbone of multi-modal large language models (MLLMs), widely used in computer vision and natural language understanding tasks. In particular, LLaMA3 models have recently been released and have achieved impressive performance in various domains with super-large scale pre-training on over 15T tokens of data. Given the wide application of low-bit quantization for LLMs in resource-constrained scenarios, we explore LLaMA3's capabilities when quantized to low bit-width. This exploration can potentially provide new insights and challenges for the low-bit quantization of LLaMA3 and other future LLMs, especially in addressing performance degradation issues that suffer in LLM compression. Specifically, we comprehensively evaluate the 10 existing post-training quantization and LoRA fine-tuning (LoRA-FT) methods of LLaMA3 on 1-8 bits and various datasets to reveal the low-bit quantization performance of LLaMA3. To uncover the capabilities of low-bit quantized MLLM, we assessed the performance of the LLaMA3-based LLaVA-Next-8B model under 2-4 ultra-low bits with post-training quantization methods. Our experimental results indicate that LLaMA3 still suffers from non-negligible degradation in linguistic and visual contexts, particularly under ultra-low bit widths. This highlights the significant performance gap at low bit-width that needs to be addressed in future developments. We expect that this empirical study will prove valuable in advancing future models, driving LLMs and MLLMs to achieve higher accuracy at lower bit to enhance practicality. Our project is released on https://github.com/Macaronlin/LLaMA3-Quantization , and quantized models are released at https://huggingface.co/Efficient-ML .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07139v1">FlexQuant: Elastic Quantization Framework for Locally Hosted LLM on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Deploying LLMs on edge devices presents serious technical challenges. Memory elasticity is crucial for edge devices with unified memory, where memory is shared and fluctuates dynamically. Existing solutions suffer from either poor transition granularity or high storage costs. We propose FlexQuant, a novel elasticity framework that generates an ensemble of quantized models, providing an elastic hosting solution with 15x granularity improvement and 10x storage reduction compared to SoTA methods. FlexQuant works with most quantization methods and creates a family of trade-off options under various storage limits through our pruning method. It brings great performance and flexibility to the edge deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19943v3">Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Mathematical reasoning tasks pose significant challenges for large language models (LLMs) because they require precise logical deduction and sequence analysis. In this work, we introduce the concept of critical tokens -- elements within reasoning trajectories that significantly influence incorrect outcomes. We present a novel framework for identifying these tokens through rollout sampling and demonstrate their substantial divergence from traditional error tokens. Through extensive experiments on datasets such as GSM8K and MATH500, we show that identifying and replacing critical tokens significantly improves model accuracy. We propose an efficient methodology for pinpointing these tokens in large-scale datasets using contrastive estimation and extend this framework to enhance model training processes with direct preference optimization (DPO). Experimental results on GSM8K and MATH500 benchmarks with the widely used models Llama-3 (8B and 70B) and Deepseek-math (7B) demonstrate the effectiveness of the proposed approach, cDPO. Our results underscore the potential of leveraging critical tokens to reduce errors in reasoning tasks, advancing the development of AI systems capable of robust logical deduction. Our code, annotated datasets, and trained models are available at https://github.com/chenzhiling9954/Critical-Tokens-Matter to support and encourage future research in this promising field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16185v3">LLM4Vuln: A Unified Evaluation Framework for Decoupling and Enhancing LLMs' Vulnerability Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-01-13
      | 💬 This is a technical report by Nanyang Technological University. Updated to support Solidity, Java and C/C++
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in various tasks, including those requiring human-level intelligence, such as vulnerability detection. However, recent efforts to use LLMs for vulnerability detection remain preliminary, as they lack a deep understanding of whether a subject LLM's vulnerability reasoning capability stems from the model itself or from external aids such as knowledge retrieval and tooling support. In this paper, we aim to decouple LLMs' vulnerability reasoning from other capabilities, such as vulnerability knowledge adoption, context information retrieval, and advanced prompt schemes. We introduce LLM4Vuln, a unified evaluation framework that separates and assesses LLMs' vulnerability reasoning capabilities and examines improvements when combined with other enhancements. We conduct controlled experiments using 147 ground-truth vulnerabilities and 147 non-vulnerable cases in Solidity, Java and C/C++, testing them in a total of 3,528 scenarios across four LLMs (GPT-3.5, GPT-4, Phi-3, and Llama 3). Our findings reveal the varying impacts of knowledge enhancement, context supplementation, and prompt schemes. We also identify 14 zero-day vulnerabilities in four pilot bug bounty programs, resulting in $3,576 in bounties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07071v1">Value Compass Leaderboard: A Platform for Fundamental and Validated Evaluation of LLMs Values</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) achieve remarkable breakthroughs, aligning their values with humans has become imperative for their responsible development and customized applications. However, there still lack evaluations of LLMs values that fulfill three desirable goals. (1) Value Clarification: We expect to clarify the underlying values of LLMs precisely and comprehensively, while current evaluations focus narrowly on safety risks such as bias and toxicity. (2) Evaluation Validity: Existing static, open-source benchmarks are prone to data contamination and quickly become obsolete as LLMs evolve. Additionally, these discriminative evaluations uncover LLMs' knowledge about values, rather than valid assessments of LLMs' behavioral conformity to values. (3) Value Pluralism: The pluralistic nature of human values across individuals and cultures is largely ignored in measuring LLMs value alignment. To address these challenges, we presents the Value Compass Leaderboard, with three correspondingly designed modules. It (i) grounds the evaluation on motivationally distinct \textit{basic values to clarify LLMs' underlying values from a holistic view; (ii) applies a \textit{generative evolving evaluation framework with adaptive test items for evolving LLMs and direct value recognition from behaviors in realistic scenarios; (iii) propose a metric that quantifies LLMs alignment with a specific value as a weighted sum over multiple dimensions, with weights determined by pluralistic values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.07058v1">Logic Meets Magic: LLMs Cracking Smart Contract Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Smart contract vulnerabilities caused significant economic losses in blockchain applications. Large Language Models (LLMs) provide new possibilities for addressing this time-consuming task. However, state-of-the-art LLM-based detection solutions are often plagued by high false-positive rates. In this paper, we push the boundaries of existing research in two key ways. First, our evaluation is based on Solidity v0.8, offering the most up-to-date insights compared to prior studies that focus on older versions (v0.4). Second, we leverage the latest five LLM models (across companies), ensuring comprehensive coverage across the most advanced capabilities in the field. We conducted a series of rigorous evaluations. Our experiments demonstrate that a well-designed prompt can reduce the false-positive rate by over 60%. Surprisingly, we also discovered that the recall rate for detecting some specific vulnerabilities in Solidity v0.8 has dropped to just 13% compared to earlier versions (i.e., v0.4). Further analysis reveals the root cause of this decline: the reliance of LLMs on identifying changes in newly introduced libraries and frameworks during detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06980v1">Combining LLM decision and RL action selection to improve RL policy for adaptive interventions</a></div>
    <div class="paper-meta">
      📅 2025-01-13
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is increasingly being used in the healthcare domain, particularly for the development of personalized health adaptive interventions. Inspired by the success of Large Language Models (LLMs), we are interested in using LLMs to update the RL policy in real time, with the goal of accelerating personalization. We use the text-based user preference to influence the action selection on the fly, in order to immediately incorporate the user preference. We use the term "user preference" as a broad term to refer to a user personal preference, constraint, health status, or a statement expressing like or dislike, etc. Our novel approach is a hybrid method that combines the LLM response and the RL action selection to improve the RL policy. Given an LLM prompt that incorporates the user preference, the LLM acts as a filter in the typical RL action selection. We investigate different prompting strategies and action selection strategies. To evaluate our approach, we implement a simulation environment that generates the text-based user preferences and models the constraints that impact behavioral dynamics. We show that our approach is able to take into account the text-based user preferences, while improving the RL policy, thus improving personalization in adaptive intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06964v1">Enhancing Patient-Centric Communication: Leveraging LLMs to Simulate Patient Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing scenarios, particularly in simulating domain-specific experts using tailored prompts. This ability enables LLMs to adopt the persona of individuals with specific backgrounds, offering a cost-effective and efficient alternative to traditional, resource-intensive user studies. By mimicking human behavior, LLMs can anticipate responses based on concrete demographic or professional profiles. In this paper, we evaluate the effectiveness of LLMs in simulating individuals with diverse backgrounds and analyze the consistency of these simulated behaviors compared to real-world outcomes. In particular, we explore the potential of LLMs to interpret and respond to discharge summaries provided to patients leaving the Intensive Care Unit (ICU). We evaluate and compare with human responses the comprehensibility of discharge summaries among individuals with varying educational backgrounds, using this analysis to assess the strengths and limitations of LLM-driven simulations. Notably, when LLMs are primed with educational background information, they deliver accurate and actionable medical guidance 88% of the time. However, when other information is provided, performance significantly drops, falling below random chance levels. This preliminary study shows the potential benefits and pitfalls of automatically generating patient-specific health information from diverse populations. While LLMs show promise in simulating health personas, our results highlight critical gaps that must be addressed before they can be reliably used in clinical settings. Our findings suggest that a straightforward query-response model could outperform a more tailored approach in delivering health information. This is a crucial first step in understanding how LLMs can be optimized for personalized health communication while maintaining accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.09172v4">Unlocking the Power of LLM Uncertainty for Active In-Context Example Selection</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance across a wide range of downstream tasks. However, it is challenging for users to discern whether the responses of LLM are generated with certainty or are fabricated to meet user expectations. In this paper, we introduce Uncertainty Tripartite Testing Paradigm (Unc-TTP), a novel method for classifying LLM uncertainty by leveraging output inconsistency. Specifically, Unc-TTP performs three rounds of sampling under varying label injection interference, enumerating all possible outcomes, and uses the degree of output inconsistency as the indicator of the LLM's intrinsic uncertainty. To validate the effectiveness of this inconsistency-defined uncertainty, we draw inspiration from Active Learning, comparing the informativeness of actively selected in-context examples. Our experiments show that uncertainty examples selected via Unc-TTP are more informative than certainty examples. Furthermore, the Unc-TTP-guided uncertainty-based active example selection strategy outperforms existing methods, highlighting its effectiveness in classifying LLM uncertainty and enhancing in-context learning. This work not only underscores the potential of inconsistency-based uncertainty classification for both open- and closed-source LLMs but also presents a practical approach for leveraging uncertainty to improve LLM performance in real-world tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.02960v2">ObfuscaTune: Obfuscated Offsite Fine-tuning and Inference of Proprietary LLMs on Private Datasets</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted at AAAI 2025 (PPAI Workshop)
    </div>
    <details class="paper-abstract">
      This work addresses the timely yet underexplored problem of performing inference and finetuning of a proprietary LLM owned by a model provider entity on the confidential/private data of another data owner entity, in a way that ensures the confidentiality of both the model and the data. Hereby, the finetuning is conducted offsite, i.e., on the computation infrastructure of a third-party cloud provider. We tackle this problem by proposing ObfuscaTune, a novel, efficient and fully utility-preserving approach that combines a simple yet effective obfuscation technique with an efficient usage of confidential computing (only 5% of the model parameters are placed on TEE). We empirically demonstrate the effectiveness of ObfuscaTune by validating it on GPT-2 models with different sizes on four NLP benchmark datasets. Finally, we compare to a na\"ive version of our approach to highlight the necessity of using random matrices with low condition numbers in our approach to reduce errors induced by the obfuscation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06842v1">SPAM: Spike-Aware Adam with Momentum Reset for Stable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance across diverse tasks, yet their training remains highly resource-intensive and susceptible to critical challenges such as training instability. A predominant source of this instability stems from gradient and loss spikes, which disrupt the learning process, often leading to costly interventions like checkpoint recovery and experiment restarts, further amplifying inefficiencies. This paper presents a comprehensive investigation into gradient spikes observed during LLM training, revealing their prevalence across multiple architectures and datasets. Our analysis shows that these spikes can be up to $1000\times$ larger than typical gradients, substantially deteriorating model performance. To address this issue, we propose Spike-Aware Adam with Momentum Reset SPAM, a novel optimizer designed to counteract gradient spikes through momentum reset and spike-aware gradient clipping. Extensive experiments, including both pre-training and fine-tuning, demonstrate that SPAM consistently surpasses Adam and its variants across various tasks, including (1) LLM pre-training from 60M to 1B, (2) 4-bit LLM pre-training,(3) reinforcement learning, and (4) Time Series Forecasting. Additionally, SPAM facilitates memory-efficient training by enabling sparse momentum, where only a subset of momentum terms are maintained and updated. When operating under memory constraints, SPAM outperforms state-of-the-art memory-efficient optimizers such as GaLore and Adam-Mini. Our work underscores the importance of mitigating gradient spikes in LLM training and introduces an effective optimization strategy that enhances both training stability and resource efficiency at scale. Code is available at https://github.com/TianjinYellow/SPAM-Optimizer.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06834v1">LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      Despite its importance, studying economic behavior across diverse, non-WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations presents significant challenges. We address this issue by introducing a novel methodology that uses Large Language Models (LLMs) to create synthetic cultural agents (SCAs) representing these populations. We subject these SCAs to classic behavioral experiments, including the dictator and ultimatum games. Our results demonstrate substantial cross-cultural variability in experimental behavior. Notably, for populations with available data, SCAs' behaviors qualitatively resemble those of real human subjects. For unstudied populations, our method can generate novel, testable hypotheses about economic behavior. By integrating AI into experimental economics, this approach offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations. Our study provides a new tool for cross-cultural economic studies and demonstrates how LLMs can help experimental behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06827v1">Leveraging Taxonomy and LLMs for Improved Multimodal Hierarchical Classification</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 11 pages, 7 figures, 2 tables, and accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      Multi-level Hierarchical Classification (MLHC) tackles the challenge of categorizing items within a complex, multi-layered class structure. However, traditional MLHC classifiers often rely on a backbone model with independent output layers, which tend to ignore the hierarchical relationships between classes. This oversight can lead to inconsistent predictions that violate the underlying taxonomy. Leveraging Large Language Models (LLMs), we propose a novel taxonomy-embedded transitional LLM-agnostic framework for multimodality classification. The cornerstone of this advancement is the ability of models to enforce consistency across hierarchical levels. Our evaluations on the MEP-3M dataset - a multi-modal e-commerce product dataset with various hierarchical levels - demonstrated a significant performance improvement compared to conventional LLM structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01174v2">Leveraging LLM and Text-Queried Separation for Noise-Robust Sound Event Detection</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted by ICASSP 2025 Workshop
    </div>
    <details class="paper-abstract">
      Sound Event Detection (SED) is challenging in noisy environments where overlapping sounds obscure target events. Language-queried audio source separation (LASS) aims to isolate the target sound events from a noisy clip. However, this approach can fail when the exact target sound is unknown, particularly in noisy test sets, leading to reduced performance. To address this issue, we leverage the capabilities of large language models (LLMs) to analyze and summarize acoustic data. By using LLMs to identify and select specific noise types, we implement a noise augmentation method for noise-robust fine-tuning. The fine-tuned model is applied to predict clip-wise event predictions as text queries for the LASS model. Our studies demonstrate that the proposed method improves SED performance in noisy environments. This work represents an early application of LLMs in noise-robust SED and suggests a promising direction for handling overlapping events in SED. Codes and pretrained models are available at https://github.com/apple-yinhan/Noise-robust-SED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06488v2">Towards Understanding Multi-Task Learning (Generalization) of LLMs via Detecting and Exploring Task-Specific Neurons</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated superior multi-task capabilities, understanding the learning mechanisms behind this is still a challenging problem. In this paper, we attempt to understand such mechanisms from the perspective of neurons. Specifically, we detect task-sensitive neurons in LLMs via gradient attribution on task-specific data. Through extensive deactivation and fine-tuning experiments, we demonstrate that the detected neurons are highly correlated with the given task, which we term as task-specific neurons. With these identified task-specific neurons, we delve into two common problems in multi-task learning and continuous learning: Generalization and Catastrophic Forgetting. We find that the overlap of task-specific neurons is strongly associated with generalization and specialization across tasks. Interestingly, at certain layers of LLMs, there is a high similarity in the parameters of different task-specific neurons, and such similarity is highly correlated with the generalization performance. Inspired by these findings, we propose a neuron-level continuous fine-tuning method that only fine-tunes the current task-specific neurons during continuous learning, and extensive experiments demonstrate the effectiveness of the proposed method. Our study provides insights into the interpretability of LLMs in multi-task learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06741v1">Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-01-12
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of large language models (LLMs) for medical applications, ensuring the reliability and accuracy of these models in clinical settings is paramount. Existing benchmarks often focus on fixed-format tasks like multiple-choice QA, which fail to capture the complexity of real-world clinical diagnostics. Moreover, traditional evaluation metrics and LLM-based evaluators struggle with misalignment, often providing oversimplified assessments that do not adequately reflect human judgment. To address these challenges, we introduce HDCEval, a Hierarchical Divide-and-Conquer Evaluation framework tailored for fine-grained alignment in medical evaluation. HDCEval is built on a set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors, encompassing Patient Question Relevance, Medical Knowledge Correctness, and Expression. The framework decomposes complex evaluation tasks into specialized subtasks, each evaluated by expert models trained through Attribute-Driven Token Optimization (ADTO) on a meticulously curated preference dataset. This hierarchical approach ensures that each aspect of the evaluation is handled with expert precision, leading to a significant improvement in alignment with human evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17596v2">LiveIdeaBench: Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Updated author list, Corrected some issues and ref
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated remarkable capabilities in scientific tasks, existing evaluation frameworks primarily assess their performance using rich contextual inputs, overlooking their ability to generate novel ideas from minimal information. We introduce LiveIdeaBench, a comprehensive benchmark that evaluates LLMs' scientific creativity and divergent thinking capabilities using single-keyword prompts. Drawing from Guilford's creativity theory, our framework employs a dynamic panel of state-of-the-art LLMs to assess generated ideas across four key dimensions: originality, feasibility, fluency, and flexibility. Through extensive experimentation with 20 leading models across 1,180 keywords spanning 18 scientific domains, we reveal that scientific creative ability shows distinct patterns from general intelligence metrics. Notably, our results demonstrate that models like QwQ-32B-preview achieve comparable creative performance to top-tier models like o1-preview, despite significant gaps in their general intelligence scores. These findings highlight the importance of specialized evaluation frameworks for scientific creativity and suggest that the development of creative capabilities in LLMs may follow different trajectories than traditional problem-solving abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06695v1">DVM: Towards Controllable LLM Agents in Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2025-01-12
      | 💬 Accepted by ICASSP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have advanced the capability of game agents in social deduction games (SDGs). These games rely heavily on conversation-driven interactions and require agents to infer, make decisions, and express based on such information. While this progress leads to more sophisticated and strategic non-player characters (NPCs) in SDGs, there exists a need to control the proficiency of these agents. This control not only ensures that NPCs can adapt to varying difficulty levels during gameplay, but also provides insights into the safety and fairness of LLM agents. In this paper, we present DVM, a novel framework for developing controllable LLM agents for SDGs, and demonstrate its implementation on one of the most popular SDGs, Werewolf. DVM comprises three main components: Predictor, Decider, and Discussor. By integrating reinforcement learning with a win rate-constrained decision chain reward mechanism, we enable agents to dynamically adjust their gameplay proficiency to achieve specified win rates. Experiments show that DVM not only outperforms existing methods in the Werewolf game, but also successfully modulates its performance levels to meet predefined win rate targets. These results pave the way for LLM agents' adaptive and balanced gameplay in SDGs, opening new avenues for research in controllable game agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06658v1">Comparing Few-Shot Prompting of GPT-4 LLMs with BERT Classifiers for Open-Response Assessment in Tutor Equity Training</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 8 Page Workshop Paper, AAAI2025 Workshop on Innovation and Responsibility in AI-Supported Education (iRAISE) - Open-response Grading, Feedback, Equity Training, LLMs, BERT, GPT-4
    </div>
    <details class="paper-abstract">
      Assessing learners in ill-defined domains, such as scenario-based human tutoring training, is an area of limited research. Equity training requires a nuanced understanding of context, but do contemporary large language models (LLMs) have a knowledge base that can navigate these nuances? Legacy transformer models like BERT, in contrast, have less real-world knowledge but can be more easily fine-tuned than commercial LLMs. Here, we study whether fine-tuning BERT on human annotations outperforms state-of-the-art LLMs (GPT-4o and GPT-4-Turbo) with few-shot prompting and instruction. We evaluate performance on four prediction tasks involving generating and explaining open-ended responses in advocacy-focused training lessons in a higher education student population learning to become middle school tutors. Leveraging a dataset of 243 human-annotated open responses from tutor training lessons, we find that BERT demonstrates superior performance using an offline fine-tuning approach, which is more resource-efficient than commercial GPT models. We conclude that contemporary GPT models may not adequately capture nuanced response patterns, especially in complex tasks requiring explanation. This work advances the understanding of AI-driven learner evaluation under the lens of fine-tuning versus few-shot prompting on the nuanced task of equity training, contributing to more effective training solutions and assisting practitioners in choosing adequate assessment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06628v1">Quantifying Relational Exploration in Cultural Heritage Knowledge Graphs with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      This paper introduces a neuro-symbolic approach for relational exploration in cultural heritage knowledge graphs, leveraging Large Language Models (LLMs) for explanation generation and a novel mathematical framework to quantify the interestingness of relationships. We demonstrate the importance of interestingness measure using a quantitative analysis, by highlighting its impact on the overall performance of our proposed system, particularly in terms of precision, recall, and F1-score. Using the Wikidata Cultural Heritage Linked Open Data (WCH-LOD) dataset, our approach yields a precision of 0.70, recall of 0.68, and an F1-score of 0.69, representing an improvement compared to graph-based (precision: 0.28, recall: 0.25, F1-score: 0.26) and knowledge-based baselines (precision: 0.45, recall: 0.42, F1-score: 0.43). Furthermore, our LLM-powered explanations exhibit better quality, reflected in BLEU (0.52), ROUGE-L (0.58), and METEOR (0.63) scores, all higher than the baseline approaches. We show a strong correlation (0.65) between interestingness measure and the quality of generated explanations, validating its effectiveness. The findings highlight the importance of LLMs and a mathematical formalization for interestingness in enhancing the effectiveness of relational exploration in cultural heritage knowledge graphs, with results that are measurable and testable. We further show that the system enables more effective exploration compared to purely knowledge-based and graph-based methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06625v1">Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 4 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in code generation tasks, yet they face significant limitations in handling complex, long-context programming challenges and demonstrating complex compositional reasoning abilities. This paper introduces a novel agentic framework for ``guided code generation'' that tries to address these limitations through a deliberately structured, fine-grained approach to code generation tasks. Our framework leverages LLMs' strengths as fuzzy searchers and approximate information retrievers while mitigating their weaknesses in long sequential reasoning and long-context understanding. Empirical evaluation using OpenAI's HumanEval benchmark with Meta's Llama 3.1 8B model (int4 precision) demonstrates a 23.79\% improvement in solution accuracy compared to direct one-shot generation. Our results indicate that structured, guided approaches to code generation can significantly enhance the practical utility of LLMs in software development while overcoming their inherent limitations in compositional reasoning and context handling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04315v2">RoRA: Efficient Fine-Tuning of LLM with Reliability Optimization for Rank Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
    </div>
    <details class="paper-abstract">
      Fine-tuning helps large language models (LLM) recover degraded information and enhance task performance. Although Low-Rank Adaptation (LoRA) is widely used and effective for fine-tuning, we have observed that its scaling factor can limit or even reduce performance as the rank size increases. To address this issue, we propose RoRA (Rank-adaptive Reliability Optimization), a simple yet effective method for optimizing LoRA's scaling factor. By replacing $\alpha/r$ with $\alpha/\sqrt{r}$, RoRA ensures improved performance as rank size increases. Moreover, RoRA enhances low-rank adaptation in fine-tuning uncompressed models and excels in the more challenging task of accuracy recovery when fine-tuning pruned models. Extensive experiments demonstrate the effectiveness of RoRA in fine-tuning both uncompressed and pruned models. RoRA surpasses the state-of-the-art (SOTA) in average accuracy and robustness on LLaMA-7B/13B, LLaMA2-7B, and LLaMA3-8B, specifically outperforming LoRA and DoRA by 6.5% and 2.9% on LLaMA-7B, respectively. In pruned model fine-tuning, RoRA shows significant advantages; for SHEARED-LLAMA-1.3, a LLaMA-7B with 81.4% pruning, RoRA achieves 5.7% higher average accuracy than LoRA and 3.9% higher than DoRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06597v1">EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 7 pages, 10 figures, 5 tables. This paper has been accepted and presented at the 2025 IEEE 15th Annual Computing and Communication Workshop and Conference (CCWC)
    </div>
    <details class="paper-abstract">
      The widespread adoption of generative AI has generated diverse opinions, with individuals expressing both support and criticism of its applications. This study investigates the emotional dynamics surrounding generative AI by analyzing human tweets referencing terms such as ChatGPT, OpenAI, Copilot, and LLMs. To further understand the emotional intelligence of ChatGPT, we examine its responses to selected tweets, highlighting differences in sentiment between human comments and LLM-generated responses. We introduce EmoXpt, a sentiment analysis framework designed to assess both human perspectives on generative AI and the sentiment embedded in ChatGPT's responses. Unlike prior studies that focus exclusively on human sentiment, EmoXpt uniquely evaluates the emotional expression of ChatGPT. Experimental results demonstrate that LLM-generated responses are notably more efficient, cohesive, and consistently positive than human responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10950v2">Understanding Multimodal LLMs: the Mechanistic Interpretability of Llava in Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Understanding the mechanisms behind Large Language Models (LLMs) is crucial for designing improved models and strategies. While recent studies have yielded valuable insights into the mechanisms of textual LLMs, the mechanisms of Multi-modal Large Language Models (MLLMs) remain underexplored. In this paper, we apply mechanistic interpretability methods to analyze the visual question answering (VQA) mechanisms in the first MLLM, Llava. We compare the mechanisms between VQA and textual QA (TQA) in color answering tasks and find that: a) VQA exhibits a mechanism similar to the in-context learning mechanism observed in TQA; b) the visual features exhibit significant interpretability when projecting the visual embeddings into the embedding space; and c) Llava enhances the existing capabilities of the corresponding textual LLM Vicuna during visual instruction tuning. Based on these findings, we develop an interpretability tool to help users and researchers identify important visual locations for final predictions, aiding in the understanding of visual hallucination. Our method demonstrates faster and more effective results compared to existing interpretability approaches. Code: \url{https://github.com/zepingyu0512/llava-mechanism}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01339v2">Improving Sequential Recommendations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 35 pages, 12 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The sequential recommendation problem has attracted considerable research attention in the past few years, leading to the rise of numerous recommendation models. In this work, we explore how Large Language Models (LLMs), which are nowadays introducing disruptive effects in many AI-based applications, can be used to build or improve sequential recommendation approaches. Specifically, we design three orthogonal approaches and hybrids of those to leverage the power of LLMs in different ways. In addition, we investigate the potential of each approach by focusing on its comprising technical aspects and determining an array of alternative choices for each one. We conduct extensive experiments on three datasets and explore a large variety of configurations, including different language models and baseline recommendation models, to obtain a comprehensive picture of the performance of each approach. Among other observations, we highlight that initializing state-of-the-art sequential recommendation models such as BERT4Rec or SASRec with embeddings obtained from an LLM can lead to substantial performance gains in terms of accuracy. Furthermore, we find that fine-tuning an LLM for recommendation tasks enables it to learn not only the tasks, but also concepts of a domain to some extent. We also show that fine-tuning OpenAI GPT leads to considerably better performance than fine-tuning Google PaLM 2. Overall, our extensive experiments indicate a huge potential value of leveraging LLMs in future recommendation approaches. We publicly share the code and data of our experiments to ensure reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16527v2">Profiling Bias in LLMs: Stereotype Dimensions in Contextual Word Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-01-11
      | 💬 Accepted to NoDaLiDa/Baltic-HLT 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are the foundation of the current successes of artificial intelligence (AI), however, they are unavoidably biased. To effectively communicate the risks and encourage mitigation efforts these models need adequate and intuitive descriptions of their discriminatory properties, appropriate for all audiences of AI. We suggest bias profiles with respect to stereotype dimensions based on dictionaries from social psychology research. Along these dimensions we investigate gender bias in contextual embeddings, across contexts and layers, and generate stereotype profiles for twelve different LLMs, demonstrating their intuition and use case for exposing and visualizing bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06471v1">The Internet of Large Language Models: An Orchestration Framework for LLM Training and Knowledge Exchange Toward Artificial General Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      This paper explores the multi-dimensional challenges faced during the development of Large Language Models (LLMs), including the massive scale of model parameters and file sizes, the complexity of development environment configuration, the singularity of model functionality, and the high costs of computational resources. To address these challenges, this paper proposes three core technical solutions: LLM sharing protocol, LLM universal environment framework, and Agent optimal path module. To solve the computational resource constraints in the early stages of research, we further innovatively propose a joint mining mechanism, achieving bilateral value sharing between computing power providers and model designers, including breakthrough rewards for optimal model paths and long-term profit distribution, thereby providing researchers with cost-optimized computational resource support and promoting the continuous development of LLM research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12236v2">Enhancing LLM Agents for Code Generation with Possibility and Pass-rate Prioritized Experience Replay</a></div>
    <div class="paper-meta">
      📅 2025-01-11
    </div>
    <details class="paper-abstract">
      Nowadays transformer-based Large Language Models (LLM) for code generation tasks usually apply sampling and filtering pipelines. Due to the sparse reward problem in code generation tasks caused by one-token incorrectness, transformer-based models will sample redundant programs till they find a correct one, leading to low efficiency. To overcome the challenge, we incorporate Experience Replay (ER) in the fine-tuning phase, where codes and programs produced are stored and will be replayed to give the LLM agent a chance to learn from past experiences. Based on the spirit of ER, we introduce a novel approach called BTP pipeline which consists of three phases: beam search sampling, testing phase, and prioritized experience replay phase. The approach makes use of failed programs collected by code models and replays programs with high Possibility and Pass-rate Prioritized value (P2Value) from the replay buffer to improve efficiency. P2Value comprehensively considers the possibility of transformers' output and pass rate and can make use of the redundant resources caused by the problem that most programs collected by LLMs fail to pass any tests. We empirically apply our approach in several LLMs, demonstrating that it enhances their performance in code generation tasks and surpasses existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06386v1">Using Pre-trained LLMs for Multivariate Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) encapsulate large amounts of knowledge and take enormous amounts of compute to train. We make use of this resource, together with the observation that LLMs are able to transfer knowledge and performance from one domain or even modality to another seemingly-unrelated area, to help with multivariate demand time series forecasting. Attention in transformer-based methods requires something worth attending to -- more than just samples of a time-series. We explore different methods to map multivariate input time series into the LLM token embedding space. In particular, our novel multivariate patching strategy to embed time series features into decoder-only pre-trained Transformers produces results competitive with state-of-the-art time series forecasting models. We also use recently-developed weight-based diagnostics to validate our findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06370v1">Towards a Probabilistic Framework for Analyzing and Improving LLM-Enabled Software</a></div>
    <div class="paper-meta">
      📅 2025-01-10
    </div>
    <details class="paper-abstract">
      Ensuring the reliability and verifiability of large language model (LLM)-enabled systems remains a significant challenge in software engineering. We propose a probabilistic framework for systematically analyzing and improving these systems by modeling and refining distributions over clusters of semantically equivalent outputs. This framework facilitates the evaluation and iterative improvement of Transference Models -- key software components that utilize LLMs to transform inputs into outputs for downstream tasks. To illustrate its utility, we apply the framework to the autoformalization problem, where natural language documentation is transformed into formal program specifications. Our case illustrates how probabilistic analysis enables the identification of weaknesses and guides focused alignment improvements, resulting in more reliable and interpretable outputs. This principled approach offers a foundation for addressing critical challenges in the development of robust LLM-enabled systems.
    </details>
</div>
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
