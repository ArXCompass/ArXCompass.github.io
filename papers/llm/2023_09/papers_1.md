# llm - 2023_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00305v1">Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method</a></div>
    <div class="paper-meta">
      📅 2023-09-30
      | 💬 Accepted by AACL 2023
    </div>
    <details class="paper-abstract">
      While large pre-trained language models (LLMs) have shown their impressive capabilities in various NLP tasks, they are still under-explored in the misinformation domain. In this paper, we examine LLMs with in-context learning (ICL) for news claim verification, and find that only with 4-shot demonstration examples, the performance of several prompting methods can be comparable with previous supervised models. To further boost performance, we introduce a Hierarchical Step-by-Step (HiSS) prompting method which directs LLMs to separate a claim into several subclaims and then verify each of them via multiple questions-answering steps progressively. Experiment results on two public misinformation datasets show that HiSS prompting outperforms state-of-the-art fully-supervised approach and strong few-shot ICL-enabled baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10694v2">"With Great Power Comes Great Responsibility!": Student and Instructor Perspectives on the influence of LLMs on Undergraduate Engineering Education</a></div>
    <div class="paper-meta">
      📅 2023-09-30
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rise in popularity of Large Language Models (LLMs) has prompted discussions in academic circles, with students exploring LLM-based tools for coursework inquiries and instructors exploring them for teaching and research. Even though a lot of work is underway to create LLM-based tools tailored for students and instructors, there is a lack of comprehensive user studies that capture the perspectives of students and instructors regarding LLMs. This paper addresses this gap by conducting surveys and interviews within undergraduate engineering universities in India. Using 1306 survey responses among students, 112 student interviews, and 27 instructor interviews around the academic usage of ChatGPT (a popular LLM), this paper offers insights into the current usage patterns, perceived benefits, threats, and challenges, as well as recommendations for enhancing the adoption of LLMs among students and instructors. These insights are further utilized to discuss the practical implications of LLMs in undergraduate engineering education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16298v2">At Which Training Stage Does Code Data Help LLMs Reasoning?</a></div>
    <div class="paper-meta">
      📅 2023-09-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable reasoning capabilities and become the foundation of language technologies. Inspired by the great success of code data in training LLMs, we naturally wonder at which training stage introducing code data can really help LLMs reasoning. To this end, this paper systematically explores the impact of code data on LLMs at different stages. Concretely, we introduce the code data at the pre-training stage, instruction-tuning stage, and both of them, respectively. Then, the reasoning capability of LLMs is comprehensively and fairly evaluated via six reasoning tasks in five domains. We critically analyze the experimental results and provide conclusions with insights. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs' general reasoning capability almost without negative transfer on other tasks. Besides, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreover, the dynamic mixing strategy of code and text data assists LLMs to learn reasoning capability step-by-step during training. These insights deepen the understanding of LLMs regarding reasoning ability for their application, such as scientific question answering, legal support, etc. The source code and model parameters are released at the link:~\url{https://github.com/yingweima2022/CodeLLM}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.01434v1">Revolutionizing Mobile Interaction: Enabling a 3 Billion Parameter GPT LLM on Mobile</a></div>
    <div class="paper-meta">
      📅 2023-09-29
    </div>
    <details class="paper-abstract">
      The field of Artificial Intelligence has witnessed remarkable progress in recent years, especially with the emergence of powerful large language models (LLMs) based on the transformer architecture. Cloud-based LLMs, such as OpenAI's ChatGPT, offer impressive capabilities but come with concerns regarding latency and privacy due to network dependencies. This article presents an innovative approach to LLM inference, envisioning a future where LLMs with billions of parameters can be executed directly on mobile devices without network connectivity. The article showcases a fine-tuned GPT LLM with 3 billion parameters that can operate smoothly on devices with as low as 4GB of memory. Through the integration of native code and model quantization techniques, the application not only serves as a general-purpose assistant but also facilitates seamless mobile interactions with text-to-actions features. The article provides insights into the training pipeline, implementation details, test results, and future directions of on-device LLM inference. This breakthrough technology opens up possibilities for empowering users with sophisticated AI capabilities while preserving their privacy and eliminating latency concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09440v3">Scope is all you need: Transforming LLMs for HPC Code</a></div>
    <div class="paper-meta">
      📅 2023-09-29
    </div>
    <details class="paper-abstract">
      With easier access to powerful compute resources, there is a growing trend in the field of AI for software development to develop larger and larger language models (LLMs) to address a variety of programming tasks. Even LLMs applied to tasks from the high-performance computing (HPC) domain are huge in size (e.g., billions of parameters) and demand expensive compute resources for training. We found this design choice confusing - why do we need large LLMs trained on natural languages and programming languages unrelated to HPC for HPC-specific tasks? In this line of work, we aim to question design choices made by existing LLMs by developing smaller LLMs for specific domains - we call them domain-specific LLMs. Specifically, we start off with HPC as a domain and propose a novel tokenizer named Tokompiler, designed specifically for preprocessing code in HPC and compilation-centric tasks. Tokompiler leverages knowledge of language primitives to generate language-oriented tokens, providing a context-aware understanding of code structure while avoiding human semantics attributed to code structures completely. We applied Tokompiler to pre-train two state-of-the-art models, SPT-Code and Polycoder, for a Fortran code corpus mined from GitHub. We evaluate the performance of these models against the conventional LLMs. Results demonstrate that Tokompiler significantly enhances code completion accuracy and semantic understanding compared to traditional tokenizers in normalized-perplexity tests, down to ~1 perplexity score. This research opens avenues for further advancements in domain-specific LLMs, catering to the unique demands of HPC and compilation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.17122v1">Benchmarking the Abilities of Large Language Models for RDF Knowledge Graph Creation and Comprehension: How Well Do LLMs Speak Turtle?</a></div>
    <div class="paper-meta">
      📅 2023-09-29
      | 💬 accepted for proceedings of DL4KG Workshop @ ISWC 2023 at ceur-ws.org
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are advancing at a rapid pace, with significant improvements at natural language processing and coding tasks. Yet, their ability to work with formal languages representing data, specifically within the realm of knowledge graph engineering, remains under-investigated. To evaluate the proficiency of various LLMs, we created a set of five tasks that probe their ability to parse, understand, analyze, and create knowledge graphs serialized in Turtle syntax. These tasks, each embodying distinct degrees of complexity and being able to scale with the size of the problem, have been integrated into our automated evaluation system, the LLM-KG-Bench. The evaluation encompassed four commercially available LLMs - GPT-3.5, GPT-4, Claude 1.3, and Claude 2.0, as well as two freely accessible offline models, GPT4All Vicuna and GPT4All Falcon 13B. This analysis offers an in-depth understanding of the strengths and shortcomings of LLMs in relation to their application within RDF knowledge graph engineering workflows utilizing Turtle representation. While our findings show that the latest commercial models outperform their forerunners in terms of proficiency with the Turtle language, they also reveal an apparent weakness. These models fall short when it comes to adhering strictly to the output formatting constraints, a crucial requirement in this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16898v1">A Sign Language Recognition System with Pepper, Lightweight-Transformer, and LLM</a></div>
    <div class="paper-meta">
      📅 2023-09-28
    </div>
    <details class="paper-abstract">
      This research explores using lightweight deep neural network architectures to enable the humanoid robot Pepper to understand American Sign Language (ASL) and facilitate non-verbal human-robot interaction. First, we introduce a lightweight and efficient model for ASL understanding optimized for embedded systems, ensuring rapid sign recognition while conserving computational resources. Building upon this, we employ large language models (LLMs) for intelligent robot interactions. Through intricate prompt engineering, we tailor interactions to allow the Pepper Robot to generate natural Co-Speech Gesture responses, laying the foundation for more organic and intuitive humanoid-robot dialogues. Finally, we present an integrated software pipeline, embodying advancements in a socially aware AI interaction model. Leveraging the Pepper Robot's capabilities, we demonstrate the practicality and effectiveness of our approach in real-world scenarios. The results highlight a profound potential for enhancing human-robot interaction through non-verbal interactions, bridging communication gaps, and making technology more accessible and understandable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09437v2">Using LLMs to Facilitate Formal Verification of RTL</a></div>
    <div class="paper-meta">
      📅 2023-09-28
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Formal property verification (FPV) has existed for decades and has been shown to be effective at finding intricate RTL bugs. However, formal properties, such as those written as SystemVerilog Assertions (SVA), are time-consuming and error-prone to write, even for experienced users. Prior work has attempted to lighten this burden by raising the abstraction level so that SVA is generated from high-level specifications. However, this does not eliminate the manual effort of reasoning and writing about the detailed hardware behavior. Motivated by the increased need for FPV in the era of heterogeneous hardware and the advances in large language models (LLMs), we set out to explore whether LLMs can capture RTL behavior and generate correct SVA properties. First, we design an FPV-based evaluation framework that measures the correctness and completeness of SVA. Then, we evaluate GPT4 iteratively to craft the set of syntax and semantic rules needed to prompt it toward creating better SVA. We extend the open-source AutoSVA framework by integrating our improved GPT4-based flow to generate safety properties, in addition to facilitating their existing flow for liveness properties. Lastly, our use cases evaluate (1) the FPV coverage of GPT4-generated SVA on complex open-source RTL and (2) using generated SVA to prompt GPT4 to create RTL from scratch. Through these experiments, we find that GPT4 can generate correct SVA even for flawed RTL, without mirroring design errors. Particularly, it generated SVA that exposed a bug in the RISC-V CVA6 core that eluded the prior work's evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16134v1">Let's Chat to Find the APIs: Connecting Human, LLM and Knowledge Graph through AI Chain</a></div>
    <div class="paper-meta">
      📅 2023-09-28
      | 💬 Accepted on ASE'2023
    </div>
    <details class="paper-abstract">
      API recommendation methods have evolved from literal and semantic keyword matching to query expansion and query clarification. The latest query clarification method is knowledge graph (KG)-based, but limitations include out-of-vocabulary (OOV) failures and rigid question templates. To address these limitations, we propose a novel knowledge-guided query clarification approach for API recommendation that leverages a large language model (LLM) guided by KG. We utilize the LLM as a neural knowledge base to overcome OOV failures, generating fluent and appropriate clarification questions and options. We also leverage the structured API knowledge and entity relationships stored in the KG to filter out noise, and transfer the optimal clarification path from KG to the LLM, increasing the efficiency of the clarification process. Our approach is designed as an AI chain that consists of five steps, each handled by a separate LLM call, to improve accuracy, efficiency, and fluency for query clarification in API recommendation. We verify the usefulness of each unit in our AI chain, which all received high scores close to a perfect 5. When compared to the baselines, our approach shows a significant improvement in MRR, with a maximum increase of 63.9% higher when the query statement is covered in KG and 37.2% when it is not. Ablation experiments reveal that the guidance of knowledge in the KG and the knowledge-guided pathfinding strategy are crucial for our approach's performance, resulting in a 19.0% and 22.2% increase in MAP, respectively. Our approach demonstrates a way to bridge the gap between KG and LLM, effectively compensating for the strengths and weaknesses of both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16031v1">DynaCon: Dynamic Robot Planner with Contextual Awareness via LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-27
      | 💬 Submitted to ICRA 2024
    </div>
    <details class="paper-abstract">
      Mobile robots often rely on pre-existing maps for effective path planning and navigation. However, when these maps are unavailable, particularly in unfamiliar environments, a different approach become essential. This paper introduces DynaCon, a novel system designed to provide mobile robots with contextual awareness and dynamic adaptability during navigation, eliminating the reliance of traditional maps. DynaCon integrates real-time feedback with an object server, prompt engineering, and navigation modules. By harnessing the capabilities of Large Language Models (LLMs), DynaCon not only understands patterns within given numeric series but also excels at categorizing objects into matched spaces. This facilitates dynamic path planner imbued with contextual awareness. We validated the effectiveness of DynaCon through an experiment where a robot successfully navigated to its goal using reasoning. Source code and experiment videos for this work can be found at: https://sites.google.com/view/dynacon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.16021v1">HuntGPT: Integrating Machine Learning-Based Anomaly Detection and Explainable AI with Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2023-09-27
    </div>
    <details class="paper-abstract">
      Machine learning (ML) is crucial in network anomaly detection for proactive threat hunting, reducing detection and response times significantly. However, challenges in model training, maintenance, and frequent false positives impact its acceptance and reliability. Explainable AI (XAI) attempts to mitigate these issues, allowing cybersecurity teams to assess AI-generated alerts with confidence, but has seen limited acceptance from incident responders. Large Language Models (LLMs) present a solution through discerning patterns in extensive information and adapting to different functional requirements. We present HuntGPT, a specialized intrusion detection dashboard applying a Random Forest classifier using the KDD99 dataset, integrating XAI frameworks like SHAP and Lime for user-friendly and intuitive model interaction, and combined with a GPT-3.5 Turbo, it delivers threats in an understandable format. The paper delves into the system's architecture, components, and technical accuracy, assessed through Certified Information Security Manager (CISM) Practice Exams, evaluating response quality across six metrics. The results demonstrate that conversational agents, supported by LLM and integrated with XAI, provide robust, explainable, and actionable AI solutions in intrusion detection, enhancing user understanding and interactive experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.15337v1">Beyond the Chat: Executable and Verifiable Text-Editing with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-27
    </div>
    <details class="paper-abstract">
      Conversational interfaces powered by Large Language Models (LLMs) have recently become a popular way to obtain feedback during document editing. However, standard chat-based conversational interfaces do not support transparency and verifiability of the editing changes that they suggest. To give the author more agency when editing with an LLM, we present InkSync, an editing interface that suggests executable edits directly within the document being edited. Because LLMs are known to introduce factual errors, Inksync also supports a 3-stage approach to mitigate this risk: Warn authors when a suggested edit introduces new information, help authors Verify the new information's accuracy through external search, and allow an auditor to perform an a-posteriori verification by Auditing the document via a trace of all auto-generated content. Two usability studies confirm the effectiveness of InkSync's components when compared to standard LLM-based chat interfaces, leading to more accurate, more efficient editing, and improved user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13701v2">ALLURE: Auditing and Improving LLM-based Evaluation of Text using Iterative In-Context-Learning</a></div>
    <div class="paper-meta">
      📅 2023-09-27
    </div>
    <details class="paper-abstract">
      From grading papers to summarizing medical documents, large language models (LLMs) are evermore used for evaluation of text generated by humans and AI alike. However, despite their extensive utility, LLMs exhibit distinct failure modes, necessitating a thorough audit and improvement of their text evaluation capabilities. Here we introduce ALLURE, a systematic approach to Auditing Large Language Models Understanding and Reasoning Errors. ALLURE involves comparing LLM-generated evaluations with annotated data, and iteratively incorporating instances of significant deviation into the evaluator, which leverages in-context learning (ICL) to enhance and improve robust evaluation of text by LLMs. Through this iterative process, we refine the performance of the evaluator LLM, ultimately reducing reliance on human annotators in the evaluation process. We anticipate ALLURE to serve diverse applications of LLMs in various domains related to evaluation of textual data, such as medical summarization, education, and and productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01463v1">Creating Trustworthy LLMs: Dealing with Hallucinations in Healthcare AI</a></div>
    <div class="paper-meta">
      📅 2023-09-26
    </div>
    <details class="paper-abstract">
      Large language models have proliferated across multiple domains in as short period of time. There is however hesitation in the medical and healthcare domain towards their adoption because of issues like factuality, coherence, and hallucinations. Give the high stakes nature of healthcare, many researchers have even cautioned against its usage until these issues are resolved. The key to the implementation and deployment of LLMs in healthcare is to make these models trustworthy, transparent (as much possible) and explainable. In this paper we describe the key elements in creating reliable, trustworthy, and unbiased models as a necessary condition for their adoption in healthcare. Specifically we focus on the quantification, validation, and mitigation of hallucinations in the context in healthcare. Lastly, we discuss how the future of LLMs in healthcare may look like.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.15840v1">How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions</a></div>
    <div class="paper-meta">
      📅 2023-09-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can "lie", which we define as outputting false statements despite "knowing" the truth in a demonstrable sense. LLMs might "lie", for example, when instructed to output misinformation. Here, we develop a simple lie detector that requires neither access to the LLM's activations (black-box) nor ground-truth knowledge of the fact in question. The detector works by asking a predefined set of unrelated follow-up questions after a suspected lie, and feeding the LLM's yes/no answers into a logistic regression classifier. Despite its simplicity, this lie detector is highly accurate and surprisingly general. When trained on examples from a single setting -- prompting GPT-3.5 to lie about factual questions -- the detector generalises out-of-distribution to (1) other LLM architectures, (2) LLMs fine-tuned to lie, (3) sycophantic lies, and (4) lies emerging in real-life scenarios such as sales. These results indicate that LLMs have distinctive lie-related behavioural patterns, consistent across architectures and contexts, which could enable general-purpose lie detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.15847v1">Disinformation Detection: An Evolving Challenge in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-25
    </div>
    <details class="paper-abstract">
      The advent of generative Large Language Models (LLMs) such as ChatGPT has catalyzed transformative advancements across multiple domains. However, alongside these advancements, they have also introduced potential threats. One critical concern is the misuse of LLMs by disinformation spreaders, leveraging these models to generate highly persuasive yet misleading content that challenges the disinformation detection system. This work aims to address this issue by answering three research questions: (1) To what extent can the current disinformation detection technique reliably detect LLM-generated disinformation? (2) If traditional techniques prove less effective, can LLMs themself be exploited to serve as a robust defense against advanced disinformation? and, (3) Should both these strategies falter, what novel approaches can be proposed to counter this burgeoning threat effectively? A holistic exploration for the formation and detection of disinformation is conducted to foster this line of research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14494v1">Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator</a></div>
    <div class="paper-meta">
      📅 2023-09-25
      | 💬 NeurIPS 2023; Project available at: https://github.com/SooLab/Free-Bloom
    </div>
    <details class="paper-abstract">
      Text-to-video is a rapidly growing research area that aims to generate a semantic, identical, and temporal coherence sequence of frames that accurately align with the input text prompt. This study focuses on zero-shot text-to-video generation considering the data- and cost-efficient. To generate a semantic-coherent video, exhibiting a rich portrayal of temporal semantics such as the whole process of flower blooming rather than a set of "moving images", we propose a novel Free-Bloom pipeline that harnesses large language models (LLMs) as the director to generate a semantic-coherence prompt sequence, while pre-trained latent diffusion models (LDMs) as the animator to generate the high fidelity frames. Furthermore, to ensure temporal and identical coherence while maintaining semantic coherence, we propose a series of annotative modifications to adapting LDMs in the reverse process, including joint noise sampling, step-aware attention shift, and dual-path interpolation. Without any video data and training requirements, Free-Bloom generates vivid and high-quality videos, awe-inspiring in generating complex scenes with semantic meaningful frame sequences. In addition, Free-Bloom is naturally compatible with LDMs-based extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14247v1">Rethinking Internet Communication Through LLMs: How Close Are We?</a></div>
    <div class="paper-meta">
      📅 2023-09-25
    </div>
    <details class="paper-abstract">
      In this paper, we rethink the way that communication among users over the Internet, one of the fundamental outcomes of the Internet evolution, takes place. Instead of users communicating directly over the Internet, we explore an architecture that enables users to communicate with (query) Large Language Models (LLMs) that capture the cognition of users on the other end of the communication channel. We present an architecture to achieve such LLM-based communication and we perform a reality check to assess how close we are today to realizing such a communication architecture from a technical point of view. Finally, we discuss several research challenges and identify interesting directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14049v1">How Novices Use LLM-Based Code Generators to Solve CS1 Coding Tasks in a Self-Paced Learning Environment</a></div>
    <div class="paper-meta">
      📅 2023-09-25
      | 💬 12 pages, Peer-Reviewed, Accepted for publication in the proceedings of the 2023 ACM Koli Calling International Conference on Computing Education Research
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain in popularity, it is important to understand how novice programmers use them. We present a thematic analysis of 33 learners, aged 10-17, independently learning Python through 45 code-authoring tasks using Codex, an LLM-based code generator. We explore several questions related to how learners used these code generators and provide an analysis of the properties of the written prompts and the generated code. Specifically, we explore (A) the context in which learners use Codex, (B) what learners are asking from Codex, (C) properties of their prompts in terms of relation to task description, language, and clarity, and prompt crafting patterns, (D) the correctness, complexity, and accuracy of the AI-generated code, and (E) how learners utilize AI-generated code in terms of placement, verification, and manual modifications. Furthermore, our analysis reveals four distinct coding approaches when writing code with an AI code generator: AI Single Prompt, where learners prompted Codex once to generate the entire solution to a task; AI Step-by-Step, where learners divided the problem into parts and used Codex to generate each part; Hybrid, where learners wrote some of the code themselves and used Codex to generate others; and Manual coding, where learners wrote the code themselves. The AI Single Prompt approach resulted in the highest correctness scores on code-authoring tasks, but the lowest correctness scores on subsequent code-modification tasks during training. Our results provide initial insight into how novice learners use AI code generators and the challenges and opportunities associated with integrating them into self-paced learning environments. We conclude with various signs of over-reliance and self-regulation, as well as opportunities for curriculum and tool development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14021v1">LORD: Low Rank Decomposition Of Monolingual Code LLMs For One-Shot Compression</a></div>
    <div class="paper-meta">
      📅 2023-09-25
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Low Rank Decomposition of matrix - splitting a large matrix into a product of two smaller matrix offers a means for compression that reduces the parameters of a model without sparsification, and hence delivering more speedup on modern hardware. Moreover, unlike quantization, the compressed linear layers remain fully differentiable and all the parameters trainable, while being able to leverage the existing highly efficient kernels over floating point matrices. We study the potential to compress Large Language Models (LLMs) for monolingual Code generation via Low Rank Decomposition (LoRD) and observe that ranks for the linear layers in these models can be reduced by upto 39.58% with less than 1% increase in perplexity. We then use Low Rank Decomposition (LoRD) to compress StarCoder 16B to 13.2B parameter with no drop and to 12.3B with minimal drop in HumanEval Pass@1 score, in less than 10 minutes on a single A100. The compressed models speeds up inference by up to 22.35% with just a single line of change in code over huggingface's implementation with pytorch backend. Low Rank Decomposition (LoRD) models remain compatible with state of the art near-lossless quantization method such as SpQR, which allows leveraging further compression gains of quantization. Lastly, QLoRA over Low Rank Decomposition (LoRD) model further reduces memory requirements by as much as 21.2% over vanilla QLoRA while offering similar gains from parameter efficient fine tuning. Our work shows Low Rank Decomposition (LoRD) as a promising new paradigm for LLM compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14389v1">Analyzing the Efficacy of an LLM-Only Approach for Image-based Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-09-25
    </div>
    <details class="paper-abstract">
      Recent document question answering models consist of two key components: the vision encoder, which captures layout and visual elements in images, and a Large Language Model (LLM) that helps contextualize questions to the image and supplements them with external world knowledge to generate accurate answers. However, the relative contributions of the vision encoder and the language model in these tasks remain unclear. This is especially interesting given the effectiveness of instruction-tuned LLMs, which exhibit remarkable adaptability to new tasks. To this end, we explore the following aspects in this work: (1) The efficacy of an LLM-only approach on document question answering tasks (2) strategies for serializing textual information within document images and feeding it directly to an instruction-tuned LLM, thus bypassing the need for an explicit vision encoder (3) thorough quantitative analysis on the feasibility of such an approach. Our comprehensive analysis encompasses six diverse benchmark datasets, utilizing LLMs of varying scales. Our findings reveal that a strategy exclusively reliant on the LLM yields results that are on par with or closely approach state-of-the-art performance across a range of datasets. We posit that this evaluation framework will serve as a guiding resource for selecting appropriate datasets for future research endeavors that emphasize the fundamental importance of layout and image content information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13574v1">LLM for Test Script Generation and Migration: Challenges, Capabilities, and Opportunities</a></div>
    <div class="paper-meta">
      📅 2023-09-24
      | 💬 Accepted by the 23rd IEEE International Conference on Software Quality, Reliability, and Security (QRS 2023)
    </div>
    <details class="paper-abstract">
      This paper investigates the application of large language models (LLM) in the domain of mobile application test script generation. Test script generation is a vital component of software testing, enabling efficient and reliable automation of repetitive test tasks. However, existing generation approaches often encounter limitations, such as difficulties in accurately capturing and reproducing test scripts across diverse devices, platforms, and applications. These challenges arise due to differences in screen sizes, input modalities, platform behaviors, API inconsistencies, and application architectures. Overcoming these limitations is crucial for achieving robust and comprehensive test automation. By leveraging the capabilities of LLMs, we aim to address these challenges and explore its potential as a versatile tool for test automation. We investigate how well LLMs can adapt to diverse devices and systems while accurately capturing and generating test scripts. Additionally, we evaluate its cross-platform generation capabilities by assessing its ability to handle operating system variations and platform-specific behaviors. Furthermore, we explore the application of LLMs in cross-app migration, where it generates test scripts across different applications and software environments based on existing scripts. Throughout the investigation, we analyze its adaptability to various user interfaces, app architectures, and interaction patterns, ensuring accurate script generation and compatibility. The findings of this research contribute to the understanding of LLMs' capabilities in test automation. Ultimately, this research aims to enhance software testing practices, empowering app developers to achieve higher levels of software quality and development efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13308v1">Calibrating LLM-Based Evaluator</a></div>
    <div class="paper-meta">
      📅 2023-09-23
      | 💬 22 pages,11 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) on language modeling and emergent capabilities make them a promising reference-free evaluator of natural language generation quality, and a competent alternative to human evaluation. However, hindered by the closed-source or high computational demand to host and tune, there is a lack of practice to further calibrate an off-the-shelf LLM-based evaluator towards better human alignment. In this work, we propose AutoCalibrate, a multi-stage, gradient-free approach to automatically calibrate and align an LLM-based evaluator toward human preference. Instead of explicitly modeling human preferences, we first implicitly encompass them within a set of human labels. Then, an initial set of scoring criteria is drafted by the language model itself, leveraging in-context learning on different few-shot examples. To further calibrate this set of criteria, we select the best performers and re-draft them with self-refinement. Our experiments on multiple text quality evaluation datasets illustrate a significant improvement in correlation with expert evaluation through calibration. Our comprehensive qualitative analysis conveys insightful intuitions and observations on the essence of effective scoring criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12938v1">Frustrated with Code Quality Issues? LLMs can Help!</a></div>
    <div class="paper-meta">
      📅 2023-09-22
    </div>
    <details class="paper-abstract">
      As software projects progress, quality of code assumes paramount importance as it affects reliability, maintainability and security of software. For this reason, static analysis tools are used in developer workflows to flag code quality issues. However, developers need to spend extra efforts to revise their code to improve code quality based on the tool findings. In this work, we investigate the use of (instruction-following) large language models (LLMs) to assist developers in revising code to resolve code quality issues. We present a tool, CORE (short for COde REvisions), architected using a pair of LLMs organized as a duo comprised of a proposer and a ranker. Providers of static analysis tools recommend ways to mitigate the tool warnings and developers follow them to revise their code. The \emph{proposer LLM} of CORE takes the same set of recommendations and applies them to generate candidate code revisions. The candidates which pass the static quality checks are retained. However, the LLM may introduce subtle, unintended functionality changes which may go un-detected by the static analysis. The \emph{ranker LLM} evaluates the changes made by the proposer using a rubric that closely follows the acceptance criteria that a developer would enforce. CORE uses the scores assigned by the ranker LLM to rank the candidate revisions before presenting them to the developer. CORE could revise 59.2% Python files (across 52 quality checks) so that they pass scrutiny by both a tool and a human reviewer. The ranker LLM is able to reduce false positives by 25.8% in these cases. CORE produced revisions that passed the static analysis tool in 76.8% Java files (across 10 quality checks) comparable to 78.3% of a specialized program repair tool, with significantly much less engineering efforts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.12499v1">CodePlan: Repository-level Coding using LLMs and Planning</a></div>
    <div class="paper-meta">
      📅 2023-09-21
    </div>
    <details class="paper-abstract">
      Software engineering activities such as package migration, fixing errors reports from static analysis or testing, and adding type annotations or other specifications to a codebase, involve pervasively editing the entire repository of code. We formulate these activities as repository-level coding tasks. Recent tools like GitHub Copilot, which are powered by Large Language Models (LLMs), have succeeded in offering high-quality solutions to localized coding problems. Repository-level coding tasks are more involved and cannot be solved directly using LLMs, since code within a repository is inter-dependent and the entire repository may be too large to fit into the prompt. We frame repository-level coding as a planning problem and present a task-agnostic framework, called CodePlan to solve it. CodePlan synthesizes a multi-step chain of edits (plan), where each step results in a call to an LLM on a code location with context derived from the entire repository, previous code changes and task-specific instructions. CodePlan is based on a novel combination of an incremental dependency analysis, a change may-impact analysis and an adaptive planning algorithm. We evaluate the effectiveness of CodePlan on two repository-level tasks: package migration (C#) and temporal code edits (Python). Each task is evaluated on multiple code repositories, each of which requires inter-dependent changes to many files (between 2-97 files). Coding tasks of this level of complexity have not been automated using LLMs before. Our results show that CodePlan has better match with the ground truth compared to baselines. CodePlan is able to get 5/6 repositories to pass the validity checks (e.g., to build without errors and make correct code edits) whereas the baselines (without planning but with the same type of contextual information as CodePlan) cannot get any of the repositories to pass them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11805v1">JobRecoGPT -- Explainable job recommendations using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-21
      | 💬 10 pages, 29 figures
    </div>
    <details class="paper-abstract">
      In today's rapidly evolving job market, finding the right opportunity can be a daunting challenge. With advancements in the field of AI, computers can now recommend suitable jobs to candidates. However, the task of recommending jobs is not same as recommending movies to viewers. Apart from must-have criteria, like skills and experience, there are many subtle aspects to a job which can decide if it is a good fit or not for a given candidate. Traditional approaches can capture the quantifiable aspects of jobs and candidates, but a substantial portion of the data that is present in unstructured form in the job descriptions and resumes is lost in the process of conversion to structured format. As of late, Large Language Models (LLMs) have taken over the AI field by storm with extraordinary performance in fields where text-based data is available. Inspired by the superior performance of LLMs, we leverage their capability to understand natural language for capturing the information that was previously getting lost during the conversion of unstructured data to structured form. To this end, we compare performance of four different approaches for job recommendations namely, (i) Content based deterministic, (ii) LLM guided, (iii) LLM unguided, and (iv) Hybrid. In this study, we present advantages and limitations of each method and evaluate their performance in terms of time requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11206v2">Retrieve-Rewrite-Answer: A KG-to-Text Enhanced LLMs Framework for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-09-21
    </div>
    <details class="paper-abstract">
      Despite their competitive performance on knowledge-intensive tasks, large language models (LLMs) still have limitations in memorizing all world knowledge especially long tail knowledge. In this paper, we study the KG-augmented language model approach for solving the knowledge graph question answering (KGQA) task that requires rich world knowledge. Existing work has shown that retrieving KG knowledge to enhance LLMs prompting can significantly improve LLMs performance in KGQA. However, their approaches lack a well-formed verbalization of KG knowledge, i.e., they ignore the gap between KG representations and textual representations. To this end, we propose an answer-sensitive KG-to-Text approach that can transform KG knowledge into well-textualized statements most informative for KGQA. Based on this approach, we propose a KG-to-Text enhanced LLMs framework for solving the KGQA task. Experiments on several KGQA benchmarks show that the proposed KG-to-Text augmented LLMs approach outperforms previous KG-augmented LLMs approaches regarding answer accuracy and usefulness of knowledge statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11688v1">LLM Guided Inductive Inference for Solving Compositional Problems</a></div>
    <div class="paper-meta">
      📅 2023-09-20
      | 💬 5 pages, ICML TEACH Workshop
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated impressive performance in question-answering tasks, their performance is limited when the questions require knowledge that is not included in the model's training data and can only be acquired through direct observation or interaction with the real world. Existing methods decompose reasoning tasks through the use of modules invoked sequentially, limiting their ability to answer deep reasoning tasks. We introduce a method, Recursion based extensible LLM (REBEL), which handles open-world, deep reasoning tasks by employing automated reasoning techniques like dynamic planning and forward-chaining strategies. REBEL allows LLMs to reason via recursive problem decomposition and utilization of external tools. The tools that REBEL uses are specified only by natural language description. We further demonstrate REBEL capabilities on a set of problems that require a deeply nested use of external tools in a compositional and conversational setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11478v1">Fictional Worlds, Real Connections: Developing Community Storytelling Social Chatbots through LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-20
    </div>
    <details class="paper-abstract">
      We address the integration of storytelling and Large Language Models (LLMs) to develop engaging and believable Social Chatbots (SCs) in community settings. Motivated by the potential of fictional characters to enhance social interactions, we introduce Storytelling Social Chatbots (SSCs) and the concept of story engineering to transform fictional game characters into "live" social entities within player communities. Our story engineering process includes three steps: (1) Character and story creation, defining the SC's personality and worldview, (2) Presenting Live Stories to the Community, allowing the SC to recount challenges and seek suggestions, and (3) Communication with community members, enabling interaction between the SC and users. We employed the LLM GPT-3 to drive our SSC prototypes, "David" and "Catherine," and evaluated their performance in an online gaming community, "DE (Alias)," on Discord. Our mixed-method analysis, based on questionnaires (N=15) and interviews (N=8) with community members, reveals that storytelling significantly enhances the engagement and believability of SCs in community settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11385v1">Safurai 001: New Qualitative Approach for Code LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2023-09-20
      | 💬 22 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      This paper presents Safurai-001, a new Large Language Model (LLM) with significant potential in the domain of coding assistance. Driven by recent advancements in coding LLMs, Safurai-001 competes in performance with the latest models like WizardCoder [Xu et al., 2023], PanguCoder [Shen et al., 2023] and Phi-1 [Gunasekar et al., 2023] but aims to deliver a more conversational interaction. By capitalizing on the progress in data engineering (including latest techniques of data transformation and prompt engineering) and instruction tuning, this new model promises to stand toe-to-toe with recent closed and open source developments. Recognizing the need for an efficacious evaluation metric for coding LLMs, this paper also introduces GPT4-based MultiParameters, an evaluation benchmark that harnesses varied parameters to present a comprehensive insight into the models functioning and performance. Our assessment shows that Safurai-001 can outperform GPT-3.5 by 1.58% and WizardCoder by 18.78% in the Code Readability parameter and more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.07992v3">Can ChatGPT Replace Traditional KBQA Models? An In-depth Analysis of the Question Answering Performance of the GPT LLM Family</a></div>
    <div class="paper-meta">
      📅 2023-09-20
      | 💬 To be published in Proceedings of ISWC 2023, 22nd International Semantic Web Conference
    </div>
    <details class="paper-abstract">
      ChatGPT is a powerful large language model (LLM) that covers knowledge resources such as Wikipedia and supports natural language question answering using its own knowledge. Therefore, there is growing interest in exploring whether ChatGPT can replace traditional knowledge-based question answering (KBQA) models. Although there have been some works analyzing the question answering performance of ChatGPT, there is still a lack of large-scale, comprehensive testing of various types of complex questions to analyze the limitations of the model. In this paper, we present a framework that follows the black-box testing specifications of CheckList proposed by Ribeiro et. al. We evaluate ChatGPT and its family of LLMs on eight real-world KB-based complex question answering datasets, which include six English datasets and two multilingual datasets. The total number of test cases is approximately 190,000. In addition to the GPT family of LLMs, we also evaluate the well-known FLAN-T5 to identify commonalities between the GPT family and other LLMs. The dataset and code are available at https://github.com/tan92hl/Complex-Question-Answering-Evaluation-of-GPT-family.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.11064v1">Exploring the Relationship between LLM Hallucinations and Prompt Linguistic Nuances: Readability, Formality, and Concreteness</a></div>
    <div class="paper-meta">
      📅 2023-09-20
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have advanced, they have brought forth new challenges, with one of the prominent issues being LLM hallucination. While various mitigation techniques are emerging to address hallucination, it is equally crucial to delve into its underlying causes. Consequently, in this preliminary exploratory investigation, we examine how linguistic factors in prompts, specifically readability, formality, and concreteness, influence the occurrence of hallucinations. Our experimental results suggest that prompts characterized by greater formality and concreteness tend to result in reduced hallucination. However, the outcomes pertaining to readability are somewhat inconclusive, showing a mixed pattern.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12477v2">GPT-3.5, GPT-4, or BARD? Evaluating LLMs Reasoning Ability in Zero-Shot Setting and Performance Boosting Through Prompts</a></div>
    <div class="paper-meta">
      📅 2023-09-19
      | 💬 Accepted for publication at Elsevier's Natural Language Processing Journal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable performance on various Natural Language Processing (NLP) tasks. However, there is a current hot debate regarding their reasoning capacity. In this paper, we examine the performance of GPT-3.5, GPT-4, and BARD models, by performing a thorough technical evaluation on different reasoning tasks across eleven distinct datasets. Our paper provides empirical evidence showcasing the superior performance of ChatGPT-4 in comparison to both ChatGPT-3.5 and BARD in zero-shot setting throughout almost all evaluated tasks. While the superiority of GPT-4 compared to GPT-3.5 might be explained by its larger size and NLP efficiency, this was not evident for BARD. We also demonstrate that the three models show limited proficiency in Inductive, Mathematical, and Multi-hop Reasoning Tasks. To bolster our findings, we present a detailed and comprehensive analysis of the results from these three models. Furthermore, we propose a set of engineered prompts that enhances the zero-shot setting performance of all three models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10544v1">Model Leeching: An Extraction Attack Targeting LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-19
    </div>
    <details class="paper-abstract">
      Model Leeching is a novel extraction attack targeting Large Language Models (LLMs), capable of distilling task-specific knowledge from a target LLM into a reduced parameter model. We demonstrate the effectiveness of our attack by extracting task capability from ChatGPT-3.5-Turbo, achieving 73% Exact Match (EM) similarity, and SQuAD EM and F1 accuracy scores of 75% and 87%, respectively for only $50 in API cost. We further demonstrate the feasibility of adversarial attack transferability from an extracted model extracted via Model Leeching to perform ML attack staging against a target LLM, resulting in an 11% increase to attack success rate when applied to ChatGPT-3.5-Turbo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10371v1">Generative AI vs. AGI: The Cognitive Strengths and Weaknesses of Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-19
    </div>
    <details class="paper-abstract">
      A moderately detailed consideration of interactive LLMs as cognitive systems is given, focusing on LLMs circa mid-2023 such as ChatGPT, GPT-4, Bard, Llama, etc.. Cognitive strengths of these systems are reviewed, and then careful attention is paid to the substantial differences between the sort of cognitive system these LLMs are, and the sort of cognitive systems human beings are. It is found that many of the practical weaknesses of these AI systems can be tied specifically to lacks in the basic cognitive architectures according to which these systems are built. It is argued that incremental improvement of such LLMs is not a viable approach to working toward human-level AGI, in practical terms given realizable amounts of compute resources. This does not imply there is nothing to learn about human-level AGI from studying and experimenting with LLMs, nor that LLMs cannot form significant parts of human-level AGI architectures that also incorporate other ideas. Social and ethical matters regarding LLMs are very briefly touched from this perspective, which implies that while care should be taken regarding misinformation and other issues, and economic upheavals will need their own social remedies based on their unpredictable course as with any powerfully impactful technology, overall the sort of policy needed as regards modern LLMs is quite different than would be the case if a more credible approximation to human-level AGI were at hand.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10294v1">Leveraging Speech PTM, Text LLM, and Emotional TTS for Speech Emotion Recognition</a></div>
    <div class="paper-meta">
      📅 2023-09-19
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      In this paper, we explored how to boost speech emotion recognition (SER) with the state-of-the-art speech pre-trained model (PTM), data2vec, text generation technique, GPT-4, and speech synthesis technique, Azure TTS. First, we investigated the representation ability of different speech self-supervised pre-trained models, and we found that data2vec has a good representation ability on the SER task. Second, we employed a powerful large language model (LLM), GPT-4, and emotional text-to-speech (TTS) model, Azure TTS, to generate emotionally congruent text and speech. We carefully designed the text prompt and dataset construction, to obtain the synthetic emotional speech data with high quality. Third, we studied different ways of data augmentation to promote the SER task with synthetic speech, including random mixing, adversarial training, transfer learning, and curriculum learning. Experiments and ablation studies on the IEMOCAP dataset demonstrate the effectiveness of our method, compared with other data augmentation methods, and data augmentation with other synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.10168v1">Few-Shot Adaptation for Parsing Contextual Utterances with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-18
      | 💬 Findings of IJCNLP-AACL 2023
    </div>
    <details class="paper-abstract">
      We evaluate the ability of semantic parsers based on large language models (LLMs) to handle contextual utterances. In real-world settings, there typically exists only a limited number of annotated contextual utterances due to annotation cost, resulting in an imbalance compared to non-contextual utterances. Therefore, parsers must adapt to contextual utterances with a few training examples. We examine four major paradigms for doing so in conversational semantic parsing i.e., Parse-with-Utterance-History, Parse-with-Reference-Program, Parse-then-Resolve, and Rewrite-then-Parse. To facilitate such cross-paradigm comparisons, we construct SMCalFlow-EventQueries, a subset of contextual examples from SMCalFlow with additional annotations. Experiments with in-context learning and fine-tuning suggest that Rewrite-then-Parse is the most promising paradigm when holistically considering parsing accuracy, annotation cost, and error types.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07686v4">Creating a Dataset for High-Performance Computing Code Translation using LLMs: A Bridge Between OpenMP Fortran and C++</a></div>
    <div class="paper-meta">
      📅 2023-09-18
      | 💬 This paper was accepted by the HPEC 2023 conference and received the Outstanding Student Paper Award
    </div>
    <details class="paper-abstract">
      In this study, we present a novel dataset for training machine learning models translating between OpenMP Fortran and C++ code. To ensure reliability and applicability, the dataset is created from a range of representative open-source OpenMP benchmarks. It is also refined using a meticulous code similarity test. The effectiveness of our dataset is assessed using both quantitative (CodeBLEU) and qualitative (human evaluation) methods. We showcase how this dataset significantly elevates the translation competencies of large language models (LLMs). Specifically, models without prior coding knowledge experienced a boost of $\mathbf{\times~5.1}$ in their CodeBLEU scores, while models with some coding familiarity saw an impressive $\mathbf{\times~9.9}$-fold increase. The best fine-tuned model using our dataset outperforms GPT-4. It is also reaching human-level accuracy. This work underscores the immense potential of our dataset in propelling advancements in the domain of code translation for high-performance computing. The dataset is accessible at \href{https://github.com/bin123apple/Fortran-CPP-HPC-code-translation-dataset}{OpenMP-Fortran-CPP-Translation}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.01105v2">A Study on the Implementation of Generative AI Services Using an Enterprise Data-Based LLM Application Architecture</a></div>
    <div class="paper-meta">
      📅 2023-09-18
    </div>
    <details class="paper-abstract">
      This study presents a method for implementing generative AI services by utilizing the Large Language Models (LLM) application architecture. With recent advancements in generative AI technology, LLMs have gained prominence across various domains. In this context, the research addresses the challenge of information scarcity and proposes specific remedies by harnessing LLM capabilities. The investigation delves into strategies for mitigating the issue of inadequate data, offering tailored solutions. The study delves into the efficacy of employing fine-tuning techniques and direct document integration to alleviate data insufficiency. A significant contribution of this work is the development of a Retrieval-Augmented Generation (RAG) model, which tackles the aforementioned challenges. The RAG model is carefully designed to enhance information storage and retrieval processes, ensuring improved content generation. The research elucidates the key phases of the information storage and retrieval methodology underpinned by the RAG model. A comprehensive analysis of these steps is undertaken, emphasizing their significance in addressing the scarcity of data. The study highlights the efficacy of the proposed method, showcasing its applicability through illustrative instances. By implementing the RAG model for information storage and retrieval, the research not only contributes to a deeper comprehension of generative AI technology but also facilitates its practical usability within enterprises utilizing LLMs. This work holds substantial value in advancing the field of generative AI, offering insights into enhancing data-driven content generation and fostering active utilization of LLM-based services within corporate settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09181v1">From Cooking Recipes to Robot Task Trees -- Improving Planning Correctness and Task Efficiency by Leveraging LLMs with a Knowledge Network</a></div>
    <div class="paper-meta">
      📅 2023-09-17
    </div>
    <details class="paper-abstract">
      Task planning for robotic cooking involves generating a sequence of actions for a robot to prepare a meal successfully. This paper introduces a novel task tree generation pipeline producing correct planning and efficient execution for cooking tasks. Our method first uses a large language model (LLM) to retrieve recipe instructions and then utilizes a fine-tuned GPT-3 to convert them into a task tree, capturing sequential and parallel dependencies among subtasks. The pipeline then mitigates the uncertainty and unreliable features of LLM outputs using task tree retrieval. We combine multiple LLM task tree outputs into a graph and perform a task tree retrieval to avoid questionable nodes and high-cost nodes to improve planning correctness and improve execution efficiency. Our evaluation results show its superior performance compared to previous works in task planning accuracy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08827v1">S3-DST: Structured Open-Domain Dialogue Segmentation and State Tracking in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-16
    </div>
    <details class="paper-abstract">
      The traditional Dialogue State Tracking (DST) problem aims to track user preferences and intents in user-agent conversations. While sufficient for task-oriented dialogue systems supporting narrow domain applications, the advent of Large Language Model (LLM)-based chat systems has introduced many real-world intricacies in open-domain dialogues. These intricacies manifest in the form of increased complexity in contextual interactions, extended dialogue sessions encompassing a diverse array of topics, and more frequent contextual shifts. To handle these intricacies arising from evolving LLM-based chat systems, we propose joint dialogue segmentation and state tracking per segment in open-domain dialogue systems. Assuming a zero-shot setting appropriate to a true open-domain dialogue system, we propose S3-DST, a structured prompting technique that harnesses Pre-Analytical Recollection, a novel grounding mechanism we designed for improving long context tracking. To demonstrate the efficacy of our proposed approach in joint segmentation and state tracking, we evaluate S3-DST on a proprietary anonymized open-domain dialogue dataset, as well as publicly available DST and segmentation datasets. Across all datasets and settings, S3-DST consistently outperforms the state-of-the-art, demonstrating its potency and robustness the next generation of LLM-based chat systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08787v1">Beyond Labels: Leveraging Deep Learning and LLMs for Content Metadata</a></div>
    <div class="paper-meta">
      📅 2023-09-15
    </div>
    <details class="paper-abstract">
      Content metadata plays a very important role in movie recommender systems as it provides valuable information about various aspects of a movie such as genre, cast, plot synopsis, box office summary, etc. Analyzing the metadata can help understand the user preferences to generate personalized recommendations and item cold starting. In this talk, we will focus on one particular type of metadata - \textit{genre} labels. Genre labels associated with a movie or a TV series help categorize a collection of titles into different themes and correspondingly setting up the audience expectation. We present some of the challenges associated with using genre label information and propose a new way of examining the genre information that we call as the \textit{Genre Spectrum}. The Genre Spectrum helps capture the various nuanced genres in a title and our offline and online experiments corroborate the effectiveness of the approach. Furthermore, we also talk about applications of LLMs in augmenting content metadata which could eventually be used to achieve effective organization of recommendations in user's 2-D home-grid.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.08210v1">Investigating Answerability of LLMs for Long-Form Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-09-15
    </div>
    <details class="paper-abstract">
      As we embark on a new era of LLMs, it becomes increasingly crucial to understand their capabilities, limitations, and differences. Toward making further progress in this direction, we strive to build a deeper understanding of the gaps between massive LLMs (e.g., ChatGPT) and smaller yet effective open-source LLMs and their distilled counterparts. To this end, we specifically focus on long-form question answering (LFQA) because it has several practical and impactful applications (e.g., troubleshooting, customer service, etc.) yet is still understudied and challenging for LLMs. We propose a question-generation method from abstractive summaries and show that generating follow-up questions from summaries of long documents can create a challenging setting for LLMs to reason and infer from long contexts. Our experimental results confirm that: (1) our proposed method of generating questions from abstractive summaries pose a challenging setup for LLMs and shows performance gaps between LLMs like ChatGPT and open-source LLMs (Alpaca, Llama) (2) open-source LLMs exhibit decreased reliance on context for generated questions from the original document, but their generation capabilities drop significantly on generated questions from summaries -- especially for longer contexts (>1024 tokens)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07755v1">Generative AI Text Classification using Ensemble LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2023-09-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance across a variety of Artificial Intelligence (AI) and natural language processing tasks, such as content creation, report generation, etc. However, unregulated malign application of these models can create undesirable consequences such as generation of fake news, plagiarism, etc. As a result, accurate detection of AI-generated language can be crucial in responsible usage of LLMs. In this work, we explore 1) whether a certain body of text is AI generated or written by human, and 2) attribution of a specific language model in generating a body of text. Texts in both English and Spanish are considered. The datasets used in this study are provided as part of the Automated Text Identification (AuTexTification) shared task. For each of the research objectives stated above, we propose an ensemble neural model that generates probabilities from different pre-trained LLMs which are used as features to a Traditional Machine Learning (TML) classifier following it. For the first task of distinguishing between AI and human generated text, our model ranked in fifth and thirteenth place (with macro $F1$ scores of 0.733 and 0.649) for English and Spanish texts, respectively. For the second task on model attribution, our model ranked in first place with macro $F1$ scores of 0.625 and 0.653 for English and Spanish texts, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.05918v3">Stochastic LLMs do not Understand Language: Towards Symbolic, Explainable and Ontologically Based LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-14
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      In our opinion the exuberance surrounding the relative success of data-driven large language models (LLMs) is slightly misguided and for several reasons (i) LLMs cannot be relied upon for factual information since for LLMs all ingested text (factual or non-factual) was created equal; (ii) due to their subsymbolic na-ture, whatever 'knowledge' these models acquire about language will always be buried in billions of microfeatures (weights), none of which is meaningful on its own; and (iii) LLMs will often fail to make the correct inferences in several linguistic contexts (e.g., nominal compounds, copredication, quantifier scope ambi-guities, intensional contexts. Since we believe the relative success of data-driven large language models (LLMs) is not a reflection on the symbolic vs. subsymbol-ic debate but a reflection on applying the successful strategy of a bottom-up reverse engineering of language at scale, we suggest in this paper applying the effective bottom-up strategy in a symbolic setting resulting in symbolic, explainable, and ontologically grounded language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07418v1">A Fast Optimization View: Reformulating Single Layer Attention in LLM Based on Tensor and SVM Trick, and Solving It in Matrix Multiplication Time</a></div>
    <div class="paper-meta">
      📅 2023-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have played a pivotal role in revolutionizing various facets of our daily existence. Solving attention regression is a fundamental task in optimizing LLMs. In this work, we focus on giving a provable guarantee for the one-layer attention network objective function $L(X,Y) = \sum_{j_0 = 1}^n \sum_{i_0 = 1}^d ( \langle \langle \exp( \mathsf{A}_{j_0} x ) , {\bf 1}_n \rangle^{-1} \exp( \mathsf{A}_{j_0} x ), A_{3} Y_{*,i_0} \rangle - b_{j_0,i_0} )^2$. Here $\mathsf{A} \in \mathbb{R}^{n^2 \times d^2}$ is Kronecker product between $A_1 \in \mathbb{R}^{n \times d}$ and $A_2 \in \mathbb{R}^{n \times d}$. $A_3$ is a matrix in $\mathbb{R}^{n \times d}$, $\mathsf{A}_{j_0} \in \mathbb{R}^{n \times d^2}$ is the $j_0$-th block of $\mathsf{A}$. The $X, Y \in \mathbb{R}^{d \times d}$ are variables we want to learn. $B \in \mathbb{R}^{n \times d}$ and $b_{j_0,i_0} \in \mathbb{R}$ is one entry at $j_0$-th row and $i_0$-th column of $B$, $Y_{*,i_0} \in \mathbb{R}^d$ is the $i_0$-column vector of $Y$, and $x \in \mathbb{R}^{d^2}$ is the vectorization of $X$. In a multi-layer LLM network, the matrix $B \in \mathbb{R}^{n \times d}$ can be viewed as the output of a layer, and $A_1= A_2 = A_3 \in \mathbb{R}^{n \times d}$ can be viewed as the input of a layer. The matrix version of $x$ can be viewed as $QK^\top$ and $Y$ can be viewed as $V$. We provide an iterative greedy algorithm to train loss function $L(X,Y)$ up $\epsilon$ that runs in $\widetilde{O}( ({\cal T}_{\mathrm{mat}}(n,n,d) + {\cal T}_{\mathrm{mat}}(n,d,d) + d^{2\omega}) \log(1/\epsilon) )$ time. Here ${\cal T}_{\mathrm{mat}}(a,b,c)$ denotes the time of multiplying $a \times b$ matrix another $b \times c$ matrix, and $\omega\approx 2.37$ denotes the exponent of matrix multiplication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.07120v1">Sight Beyond Text: Multi-Modal Training Enhances LLMs in Truthfulness and Ethics</a></div>
    <div class="paper-meta">
      📅 2023-09-13
    </div>
    <details class="paper-abstract">
      Multi-modal large language models (MLLMs) are trained based on large language models (LLM), with an enhanced capability to comprehend multi-modal inputs and generate textual responses. While they excel in multi-modal tasks, the pure NLP abilities of MLLMs are often underestimated and left untested. In this study, we get out of the box and unveil an intriguing characteristic of MLLMs -- our preliminary results suggest that visual instruction tuning, a prevailing strategy for transitioning LLMs into MLLMs, unexpectedly and interestingly helps models attain both improved truthfulness and ethical alignment in the pure NLP context. For example, a visual-instruction-tuned LLaMA2 7B model surpasses the performance of the LLaMA2-chat 7B model, fine-tuned with over one million human annotations, on TruthfulQA-mc and Ethics benchmarks. Further analysis reveals that the improved alignment can be attributed to the superior instruction quality inherent to visual-text data. In releasing our code at github.com/UCSC-VLAA/Sight-Beyond-Text, we aspire to foster further exploration into the intrinsic value of visual-text synergies and, in a broader scope, multi-modal interactions in alignment research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.06541v1">Text Encoders Lack Knowledge: Leveraging Generative LLMs for Domain-Specific Semantic Textual Similarity</a></div>
    <div class="paper-meta">
      📅 2023-09-12
      | 💬 Under review GEM@EMNLP-2023, 12 pages
    </div>
    <details class="paper-abstract">
      Amidst the sharp rise in the evaluation of large language models (LLMs) on various tasks, we find that semantic textual similarity (STS) has been under-explored. In this study, we show that STS can be cast as a text generation problem while maintaining strong performance on multiple STS benchmarks. Additionally, we show generative LLMs significantly outperform existing encoder-based STS models when characterizing the semantic similarity between two texts with complex semantic relationships dependent on world knowledge. We validate this claim by evaluating both generative LLMs and existing encoder-based STS models on three newly collected STS challenge sets which require world knowledge in the domains of Health, Politics, and Sports. All newly collected data is sourced from social media content posted after May 2023 to ensure the performance of closed-source models like ChatGPT cannot be credited to memorization. Our results show that, on average, generative LLMs outperform the best encoder-only baselines by an average of 22.3% on STS tasks requiring world knowledge. Our results suggest generative language models with STS-specific prompting strategies achieve state-of-the-art performance in complex, domain-specific STS tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.04564v1">When Less is More: Investigating Data Pruning for Pretraining LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2023-09-08
      | 💬 14 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large volumes of text data have contributed significantly to the development of large language models (LLMs) in recent years. This data is typically acquired by scraping the internet, leading to pretraining datasets comprised of noisy web text. To date, efforts to prune these datasets down to a higher quality subset have relied on hand-crafted heuristics encoded as rule-based filters. In this work, we take a wider view and explore scalable estimates of data quality that can be used to systematically measure the quality of pretraining data. We perform a rigorous comparison at scale of the simple data quality estimator of perplexity, as well as more sophisticated and computationally intensive estimates of the Error L2-Norm and memorization. These metrics are used to rank and prune pretraining corpora, and we subsequently compare LLMs trained on these pruned datasets. Surprisingly, we find that the simple technique of perplexity outperforms our more computationally expensive scoring methods. We improve over our no-pruning baseline while training on as little as 30% of the original training dataset. Our work sets the foundation for unexplored strategies in automatically curating high quality corpora and suggests the majority of pretraining data can be removed while retaining performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.04369v1">Beyond Static Datasets: A Deep Interaction Approach to LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2023-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made progress in various real-world tasks, which stimulates requirements for the evaluation of LLMs. Existing LLM evaluation methods are mainly supervised signal-based which depends on static datasets and cannot evaluate the ability of LLMs in dynamic real-world scenarios where deep interaction widely exists. Other LLM evaluation methods are human-based which are costly and time-consuming and are incapable of large-scale evaluation of LLMs. To address the issues above, we propose a novel Deep Interaction-based LLM-evaluation framework. In our proposed framework, LLMs' performances in real-world domains can be evaluated from their deep interaction with other LLMs in elaborately designed evaluation tasks. Furthermore, our proposed framework is a general evaluation method that can be applied to a host of real-world tasks such as machine translation and code generation. We demonstrate the effectiveness of our proposed method through extensive experiments on four elaborately designed evaluation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.16797v2">Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2023-09-08
      | 💬 DSTC11 best paper for Track 4
    </div>
    <details class="paper-abstract">
      Despite significant research effort in the development of automatic dialogue evaluation metrics, little thought is given to evaluating dialogues other than in English. At the same time, ensuring metrics are invariant to semantically similar responses is also an overlooked topic. In order to achieve the desired properties of robustness and multilinguality for dialogue evaluation metrics, we propose a novel framework that takes advantage of the strengths of current evaluation models with the newly-established paradigm of prompting Large Language Models (LLMs). Empirical results show our framework achieves state of the art results in terms of mean Spearman correlation scores across several benchmarks and ranks first place on both the Robust and Multilingual tasks of the DSTC11 Track 4 "Automatic Evaluation Metrics for Open-Domain Dialogue Systems", proving the evaluation capabilities of prompted LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.03876v1">OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-07
      | 💬 6 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 different biases (political, geographic, gender, age) and derived an instruction-tuning corpus in which each answer was written by members of one of these demographics. This paper presents OpinionGPT, illustrates how we trained the bias-aware model and showcases the web application (available at https://opiniongpt.informatik.hu-berlin.de).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09183v2">RatGPT: Turning online LLMs into Proxies for Malware Attacks</a></div>
    <div class="paper-meta">
      📅 2023-09-07
    </div>
    <details class="paper-abstract">
      The evolution of Generative AI and the capabilities of the newly released Large Language Models (LLMs) open new opportunities in software engineering. However, they also lead to new challenges in cybersecurity. Recently, researchers have shown the possibilities of using LLMs such as ChatGPT to generate malicious content that can directly be exploited or guide inexperienced hackers to weaponize tools and code. These studies covered scenarios that still require the attacker to be in the middle of the loop. In this study, we leverage openly available plugins and use an LLM as proxy between the attacker and the victim. We deliver a proof-of-concept where ChatGPT is used for the dissemination of malicious software while evading detection, alongside establishing the communication to a command and control (C2) server to receive commands to interact with a victim's system. Finally, we present the general approach as well as essential elements in order to stay undetected and make the attack a success. This proof-of-concept highlights significant cybersecurity issues with openly available plugins and LLMs, which require the development of security guidelines, controls, and mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.03118v1">Knowledge Solver: Teaching LLMs to Search for Domain Knowledge from Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2023-09-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), such as ChatGPT and GPT-4, are versatile and can solve different tasks due to their emergent ability and generalizability. However, LLMs sometimes lack domain-specific knowledge to perform tasks, which would also cause hallucination during inference. In some previous works, additional modules like graph neural networks (GNNs) are trained on retrieved knowledge from external knowledge bases, aiming to mitigate the problem of lacking domain-specific knowledge. However, incorporating additional modules: 1) would need retraining additional modules when encountering novel domains; 2) would become a bottleneck since LLMs' strong abilities are not fully utilized for retrieval. In this paper, we propose a paradigm, termed Knowledge Solver (KSL), to teach LLMs to search for essential knowledge from external knowledge bases by harnessing their own strong generalizability. Specifically, we design a simple yet effective prompt to transform retrieval into a multi-hop decision sequence, which empowers LLMs with searching knowledge ability in zero-shot manner. Additionally, KSL is able to provide complete retrieval paths and therefore increase explainability of LLMs' reasoning processes. We conduct experiments on three datasets: CommonsenseQA, OpenbookQA, and MedQA-USMLE, and found that our approach improves LLM baseline performance by a relatively large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.03044v1">Method-Level Bug Severity Prediction using Source Code Metrics and LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-06
    </div>
    <details class="paper-abstract">
      In the past couple of decades, significant research efforts are devoted to the prediction of software bugs. However, most existing work in this domain treats all bugs the same, which is not the case in practice. It is important for a defect prediction method to estimate the severity of the identified bugs so that the higher-severity ones get immediate attention. In this study, we investigate source code metrics, source code representation using large language models (LLMs), and their combination in predicting bug severity labels of two prominent datasets. We leverage several source metrics at method-level granularity to train eight different machine-learning models. Our results suggest that Decision Tree and Random Forest models outperform other models regarding our several evaluation metrics. We then use the pre-trained CodeBERT LLM to study the source code representations' effectiveness in predicting bug severity. CodeBERT finetuning improves the bug severity prediction results significantly in the range of 29%-140% for several evaluation metrics, compared to the best classic prediction model on source code metric. Finally, we integrate source code metrics into CodeBERT as an additional input, using our two proposed architectures, which both enhance the CodeBERT model effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.13387v2">Do-Not-Answer: A Dataset for Evaluating Safeguards in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-04
      | 💬 18 pages, 9 figures, 11 tables
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLMs), new and hard-to-predict harmful capabilities are emerging. This requires developers to be able to identify risks through the evaluation of "dangerous capabilities" in order to responsibly deploy LLMs. In this work, we collect the first open-source dataset to evaluate safeguards in LLMs, and deploy safer open-source LLMs at a low cost. Our dataset is curated and filtered to consist only of instructions that responsible language models should not follow. We annotate and assess the responses of six popular LLMs to these instructions. Based on our annotation, we proceed to train several BERT-like classifiers, and find that these small classifiers can achieve results that are comparable with GPT-4 on automatic safety evaluation. Warning: this paper contains example data that may be offensive, harmful, or biased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.01172v1">FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs</a></div>
    <div class="paper-meta">
      📅 2023-09-03
    </div>
    <details class="paper-abstract">
      The rapid growth of memory and computation requirements of large language models (LLMs) has outpaced the development of hardware, hindering people who lack large-scale high-end GPUs from training or deploying LLMs. However, consumer-level GPUs, which constitute a larger market share, are typically overlooked in LLM due to their weaker computing performance, smaller storage capacity, and lower communication bandwidth. Additionally, users may have privacy concerns when interacting with remote LLMs. In this paper, we envision a decentralized system unlocking the potential vast untapped consumer-level GPUs in pre-training, inference and fine-tuning of LLMs with privacy protection. However, this system faces critical challenges, including limited CPU and GPU memory, low network bandwidth, the variability of peer and device heterogeneity. To address these challenges, our system design incorporates: 1) a broker with backup pool to implement dynamic join and quit of computing providers; 2) task scheduling with hardware performance to improve system efficiency; 3) abstracting ML procedures into directed acyclic graphs (DAGs) to achieve model and task universality; 4) abstracting intermediate represention and execution planes to ensure compatibility of various devices and deep learning (DL) frameworks. Our performance analysis demonstrates that 50 RTX 3080 GPUs can achieve throughputs comparable to those of 4 H100 GPUs, which are significantly more expensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.00841v1">LeanContext: Cost-Efficient Domain-Specific Question Answering Using LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-02
      | 💬 The paper is under review
    </div>
    <details class="paper-abstract">
      Question-answering (QA) is a significant application of Large Language Models (LLMs), shaping chatbot capabilities across healthcare, education, and customer service. However, widespread LLM integration presents a challenge for small businesses due to the high expenses of LLM API usage. Costs rise rapidly when domain-specific data (context) is used alongside queries for accurate domain-specific LLM responses. One option is to summarize the context by using LLMs and reduce the context. However, this can also filter out useful information that is necessary to answer some domain-specific queries. In this paper, we shift from human-oriented summarizers to AI model-friendly summaries. Our approach, LeanContext, efficiently extracts $k$ key sentences from the context that are closely aligned with the query. The choice of $k$ is neither static nor random; we introduce a reinforcement learning technique that dynamically determines $k$ based on the query and context. The rest of the less important sentences are reduced using a free open source text reduction method. We evaluate LeanContext against several recent query-aware and query-unaware context reduction approaches on prominent datasets (arxiv papers and BBC news articles). Despite cost reductions of $37.29\%$ to $67.81\%$, LeanContext's ROUGE-1 score decreases only by $1.41\%$ to $2.65\%$ compared to a baseline that retains the entire context (no summarization). Additionally, if free pretrained LLM-based summarizers are used to reduce context (into human consumable summaries), LeanContext can further modify the reduced context to enhance the accuracy (ROUGE-1 score) by $13.22\%$ to $24.61\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.00667v1">Taken out of context: On measuring situational awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-09-01
    </div>
    <details class="paper-abstract">
      We aim to better understand the emergence of `situational awareness' in large language models (LLMs). A model is situationally aware if it's aware that it's a model and can recognize whether it's currently in testing or deployment. Today's LLMs are tested for safety and alignment before they are deployed. An LLM could exploit situational awareness to achieve a high score on safety tests, while taking harmful actions after deployment. Situational awareness may emerge unexpectedly as a byproduct of model scaling. One way to better foresee this emergence is to run scaling experiments on abilities necessary for situational awareness. As such an ability, we propose `out-of-context reasoning' (in contrast to in-context learning). We study out-of-context reasoning experimentally. First, we finetune an LLM on a description of a test while providing no examples or demonstrations. At test time, we assess whether the model can pass the test. To our surprise, we find that LLMs succeed on this out-of-context reasoning task. Their success is sensitive to the training setup and only works when we apply data augmentation. For both GPT-3 and LLaMA-1, performance improves with model size. These findings offer a foundation for further empirical study, towards predicting and potentially controlling the emergence of situational awareness in LLMs. Code is available at: https://github.com/AsaCooperStickland/situational-awareness-evals.
    </details>
</div>
