# llm - 2023_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.09991v3">Supporting Human-AI Collaboration in Auditing LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-30
      | 💬 21 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models are becoming increasingly pervasive and ubiquitous in society via deployment in sociotechnical systems. Yet these language models, be it for classification or generation, have been shown to be biased and behave irresponsibly, causing harm to people at scale. It is crucial to audit these language models rigorously. Existing auditing tools leverage either or both humans and AI to find failures. In this work, we draw upon literature in human-AI collaboration and sensemaking, and conduct interviews with research experts in safe and fair AI, to build upon the auditing tool: AdaTest (Ribeiro and Lundberg, 2022), which is powered by a generative large language model (LLM). Through the design process we highlight the importance of sensemaking and human-AI communication to leverage complementary strengths of humans and generative models in collaborative auditing. To evaluate the effectiveness of the augmented tool, AdaTest++, we conduct user studies with participants auditing two commercial language models: OpenAI's GPT-3 and Azure's sentiment analysis model. Qualitative analysis shows that AdaTest++ effectively leverages human strengths such as schematization, hypothesis formation and testing. Further, with our tool, participants identified a variety of failures modes, covering 26 different topics over 2 tasks, that have been shown before in formal audits and also those previously under-reported.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18445v1">VTimeLLM: Empower LLM to Grasp Video Moments</a></div>
    <div class="paper-meta">
      📅 2023-11-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable text understanding capabilities, which have been extended as Video LLMs to handle video data for comprehending visual details. However, existing Video LLMs can only provide a coarse description of the entire video, failing to capture the precise start and end time boundary of specific events. In this paper, we solve this issue via proposing VTimeLLM, a novel Video LLM designed for fine-grained video moment understanding and reasoning with respect to time boundary. Specifically, our VTimeLLM adopts a boundary-aware three-stage training strategy, which respectively utilizes image-text pairs for feature alignment, multiple-event videos to increase temporal-boundary awareness, and high-quality video-instruction tuning to further improve temporal understanding ability as well as align with human intents. Extensive experiments demonstrate that in fine-grained time-related comprehension tasks for videos such as Temporal Video Grounding and Dense Video Captioning, VTimeLLM significantly outperforms existing Video LLMs. Besides, benefits from the fine-grained temporal understanding of the videos further enable VTimeLLM to beat existing Video LLMs in video dialogue benchmark, showing its superior cross-modal understanding and reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18251v1">Can Large Language Models Be Good Companions? An LLM-Based Eyewear System with Conversational Common Ground</a></div>
    <div class="paper-meta">
      📅 2023-11-30
      | 💬 36 pages, 25 figures, Under review at ACM IMWUT
    </div>
    <details class="paper-abstract">
      Developing chatbots as personal companions has long been a goal of artificial intelligence researchers. Recent advances in Large Language Models (LLMs) have delivered a practical solution for endowing chatbots with anthropomorphic language capabilities. However, it takes more than LLMs to enable chatbots that can act as companions. Humans use their understanding of individual personalities to drive conversations. Chatbots also require this capability to enable human-like companionship. They should act based on personalized, real-time, and time-evolving knowledge of their owner. We define such essential knowledge as the \textit{common ground} between chatbots and their owners, and we propose to build a common-ground-aware dialogue system from an LLM-based module, named \textit{OS-1}, to enable chatbot companionship. Hosted by eyewear, OS-1 can sense the visual and audio signals the user receives and extract real-time contextual semantics. Those semantics are categorized and recorded to formulate historical contexts from which the user's profile is distilled and evolves over time, i.e., OS-1 gradually learns about its user. OS-1 combines knowledge from real-time semantics, historical contexts, and user-specific profiles to produce a common-ground-aware prompt input into the LLM module. The LLM's output is converted to audio, spoken to the wearer when appropriate.We conduct laboratory and in-field studies to assess OS-1's ability to build common ground between the chatbot and its user. The technical feasibility and capabilities of the system are also evaluated. OS-1, with its common-ground awareness, can significantly improve user satisfaction and potentially lead to downstream tasks such as personal emotional support and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.16153v2">Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2023-11-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08592v2">AART: AI-Assisted Red-Teaming with Diverse Data Generation for New LLM-powered Applications</a></div>
    <div class="paper-meta">
      📅 2023-11-29
    </div>
    <details class="paper-abstract">
      Adversarial testing of large language models (LLMs) is crucial for their safe and responsible deployment. We introduce a novel approach for automated generation of adversarial evaluation datasets to test the safety of LLM generations on new downstream applications. We call it AI-assisted Red-Teaming (AART) - an automated alternative to current manual red-teaming efforts. AART offers a data generation and augmentation pipeline of reusable and customizable recipes that reduce human effort significantly and enable integration of adversarial testing earlier in new product development. AART generates evaluation datasets with high diversity of content characteristics critical for effective adversarial testing (e.g. sensitive and harmful concepts, specific to a wide range of cultural and geographic regions and application scenarios). The data generation is steered by AI-assisted recipes to define, scope and prioritize diversity within the application context. This feeds into a structured LLM-generation process that scales up evaluation priorities. Compared to some state-of-the-art tools, AART shows promising results in terms of concept coverage and data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.15065v2">Synergizing Human-AI Agency: A Guide of 23 Heuristics for Service Co-Creation with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-29
      | 💬 V1.0 on Oct 25th, 2023
    </div>
    <details class="paper-abstract">
      This empirical study serves as a primer for interested service providers to determine if and how Large Language Models (LLMs) technology will be integrated for their practitioners and the broader community. We investigate the mutual learning journey of non-AI experts and AI through CoAGent, a service co-creation tool with LLM-based agents. Engaging in a three-stage participatory design processes, we work with with 23 domain experts from public libraries across the U.S., uncovering their fundamental challenges of integrating AI into human workflows. Our findings provide 23 actionable "heuristics for service co-creation with AI", highlighting the nuanced shared responsibilities between humans and AI. We further exemplar 9 foundational agency aspects for AI, emphasizing essentials like ownership, fair treatment, and freedom of expression. Our innovative approach enriches the participatory design model by incorporating AI as crucial stakeholders and utilizing AI-AI interaction to identify blind spots. Collectively, these insights pave the way for synergistic and ethical human-AI co-creation in service contexts, preparing for workforce ecosystems where AI coexists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.18034v1">Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings</a></div>
    <div class="paper-meta">
      📅 2023-11-29
    </div>
    <details class="paper-abstract">
      Cross-lingual transfer learning is an important property of multilingual large language models (LLMs). But how do LLMs represent relationships between languages? Every language model has an input layer that maps tokens to vectors. This ubiquitous layer of language models is often overlooked. We find that similarities between these input embeddings are highly interpretable and that the geometry of these embeddings differs between model families. In one case (XLM-RoBERTa), embeddings encode language: tokens in different writing systems can be linearly separated with an average of 99.2% accuracy. Another family (mT5) represents cross-lingual semantic similarity: the 50 nearest neighbors for any token represent an average of 7.61 writing systems, and are frequently translations. This result is surprising given that there is no explicit parallel cross-lingual training corpora and no explicit incentive for translations in pre-training objectives. Our research opens the door for investigations in 1) The effect of pre-training and model architectures on representations of languages and 2) The applications of cross-lingual representations embedded in language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17487v1">Taiwan LLM: Bridging the Linguistic Divide with a Culturally Aligned Language Model</a></div>
    <div class="paper-meta">
      📅 2023-11-29
    </div>
    <details class="paper-abstract">
      In the realm of language models, the nuanced linguistic and cultural intricacies of Traditional Chinese, as spoken in Taiwan, have been largely overlooked. This paper introduces Taiwan LLM, a pioneering Large Language Model that specifically caters to the Traditional Chinese language, with a focus on the variant used in Taiwan. Leveraging a comprehensive pretraining corpus and instruction-finetuning datasets, we have developed a model that not only understands the complexities of Traditional Chinese but also embodies the cultural context of Taiwan. Taiwan LLM represents the first of its kind, a model that is not only linguistically accurate but also culturally resonant with its user base. Our evaluations demonstrate that Taiwan LLM achieves superior performance in understanding and generating Traditional Chinese text, outperforming existing models that are predominantly trained on Simplified Chinese or English. The open-source release of Taiwan LLM invites collaboration and further innovation, ensuring that the linguistic diversity of Chinese speakers is embraced and well-served. The model, datasets, and further resources are made publicly available to foster ongoing research and development in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00055v1">LEAP: LLM-Generation of Egocentric Action Programs</a></div>
    <div class="paper-meta">
      📅 2023-11-29
      | 💬 Dataset: https://drive.google.com/drive/folders/1Cpkw_TI1IIxXdzor0pOXG3rWJWuKU5Ex?usp=drive_link
    </div>
    <details class="paper-abstract">
      We introduce LEAP (illustrated in Figure 1), a novel method for generating video-grounded action programs through use of a Large Language Model (LLM). These action programs represent the motoric, perceptual, and structural aspects of action, and consist of sub-actions, pre- and post-conditions, and control flows. LEAP's action programs are centered on egocentric video and employ recent developments in LLMs both as a source for program knowledge and as an aggregator and assessor of multimodal video information. We apply LEAP over a majority (87\%) of the training set of the EPIC Kitchens dataset, and release the resulting action programs as a publicly available dataset here (https://drive.google.com/drive/folders/1Cpkw_TI1IIxXdzor0pOXG3rWJWuKU5Ex?usp=drive_link). We employ LEAP as a secondary source of supervision, using its action programs in a loss term applied to action recognition and anticipation networks. We demonstrate sizable improvements in performance in both tasks due to training with the LEAP dataset. Our method achieves 1st place on the EPIC Kitchens Action Recognition leaderboard as of November 17 among the networks restricted to RGB-input (see Supplementary Materials).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16153v2">Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2023-11-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.16153v2">Identifying and Mitigating Vulnerabilities in LLM-Integrated Applications</a></div>
    <div class="paper-meta">
      📅 2023-11-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as the service backend for LLM-integrated applications such as code completion and AI-powered search. LLM-integrated applications serve as middleware to refine users' queries with domain-specific knowledge to better inform LLMs and enhance the responses. Despite numerous opportunities and benefits, LLM-integrated applications also introduce new attack surfaces. Understanding, minimizing, and eliminating these emerging attack surfaces is a new area of research. In this work, we consider a setup where the user and LLM interact via an LLM-integrated application in the middle. We focus on the communication rounds that begin with user's queries and end with LLM-integrated application returning responses to the queries, powered by LLMs at the service backend. For this query-response protocol, we identify potential vulnerabilities that can originate from the malicious application developer or from an outsider threat initiator that is able to control the database access, manipulate and poison data that are high-risk for the user. Successful exploits of the identified vulnerabilities result in the users receiving responses tailored to the intent of a threat initiator. We assess such threats against LLM-integrated applications empowered by OpenAI GPT-3.5 and GPT-4. Our empirical results show that the threats can effectively bypass the restrictions and moderation policies of OpenAI, resulting in users receiving responses that contain bias, toxic content, privacy risk, and disinformation. To mitigate those threats, we identify and define four key properties, namely integrity, source identification, attack detectability, and utility preservation, that need to be satisfied by a safe LLM-integrated application. Based on these properties, we develop a lightweight, threat-agnostic defense that mitigates both insider and outsider threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17107v1">ClimateX: Do LLMs Accurately Assess Human Expert Confidence in Climate Statements?</a></div>
    <div class="paper-meta">
      📅 2023-11-28
      | 💬 Tackling Climate Change with Machine Learning workshop at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Evaluating the accuracy of outputs generated by Large Language Models (LLMs) is especially important in the climate science and policy domain. We introduce the Expert Confidence in Climate Statements (ClimateX) dataset, a novel, curated, expert-labeled dataset consisting of 8094 climate statements collected from the latest Intergovernmental Panel on Climate Change (IPCC) reports, labeled with their associated confidence levels. Using this dataset, we show that recent LLMs can classify human expert confidence in climate-related statements, especially in a few-shot learning setting, but with limited (up to 47%) accuracy. Overall, models exhibit consistent and significant over-confidence on low and medium confidence statements. We highlight implications of our results for climate communication, LLMs evaluation strategies, and the use of LLMs in information retrieval systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09919v3">Plug in the Safety Chip: Enforcing Constraints for LLM-driven Robot Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-28
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled a new research domain, LLM agents, for solving robotics and planning tasks by leveraging the world knowledge and general reasoning abilities of LLMs obtained during pretraining. However, while considerable effort has been made to teach the robot the "dos," the "don'ts" received relatively less attention. We argue that, for any practical usage, it is as crucial to teach the robot the "don'ts": conveying explicit instructions about prohibited actions, assessing the robot's comprehension of these restrictions, and, most importantly, ensuring compliance. Moreover, verifiable safe operation is essential for deployments that satisfy worldwide standards such as ISO 61508, which defines standards for safely deploying robots in industrial factory environments worldwide. Aiming at deploying the LLM agents in a collaborative environment, we propose a queryable safety constraint module based on linear temporal logic (LTL) that simultaneously enables natural language (NL) to temporal constraints encoding, safety violation reasoning and explaining, and unsafe action pruning. To demonstrate the effectiveness of our system, we conducted experiments in VirtualHome environment and on a real robot. The experimental results show that our system strictly adheres to the safety constraints and scales well with complex safety constraints, highlighting its potential for practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.15759v1">Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-28
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual-language understanding, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance overall capabilities of LLMs, which could be regraded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce the Modular Visual Memory, a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixtures-of-Multimodal Experts architecture in LLMs to invoke multimodal knowledge collaboration during generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on multimodal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16101v1">How Many Unicorns Are in This Image? A Safety Evaluation Benchmark for Vision LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-27
      | 💬 H.T., C.C., and Z.W. contribute equally. Work done during H.T. and Z.W.'s internship at UCSC, and C.C. and Y.Z.'s internship at UNC
    </div>
    <details class="paper-abstract">
      This work focuses on the potential of Vision LLMs (VLLMs) in visual reasoning. Different from prior studies, we shift our focus from evaluating standard performance to introducing a comprehensive safety evaluation suite, covering both out-of-distribution (OOD) generalization and adversarial robustness. For the OOD evaluation, we present two novel VQA datasets, each with one variant, designed to test model performance under challenging conditions. In exploring adversarial robustness, we propose a straightforward attack strategy for misleading VLLMs to produce visual-unrelated responses. Moreover, we assess the efficacy of two jailbreaking strategies, targeting either the vision or language component of VLLMs. Our evaluation of 21 diverse models, ranging from open-source VLLMs to GPT-4V, yields interesting observations: 1) Current VLLMs struggle with OOD texts but not images, unless the visual information is limited; and 2) These VLLMs can be easily misled by deceiving vision encoders only, and their vision-language training often compromise safety protocols. We release this safety evaluation suite at https://github.com/UCSC-VLAA/vllm-safety-benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16090v1">Self-correcting LLM-controlled Diffusion Models</a></div>
    <div class="paper-meta">
      📅 2023-11-27
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Text-to-image generation has witnessed significant progress with the advent of diffusion models. Despite the ability to generate photorealistic images, current text-to-image diffusion models still often struggle to accurately interpret and follow complex input text prompts. In contrast to existing models that aim to generate images only with their best effort, we introduce Self-correcting LLM-controlled Diffusion (SLD). SLD is a framework that generates an image from the input prompt, assesses its alignment with the prompt, and performs self-corrections on the inaccuracies in the generated image. Steered by an LLM controller, SLD turns text-to-image generation into an iterative closed-loop process, ensuring correctness in the resulting image. SLD is not only training-free but can also be seamlessly integrated with diffusion models behind API access, such as DALL-E 3, to further boost the performance of state-of-the-art diffusion models. Experimental results show that our approach can rectify a majority of incorrect generations, particularly in generative numeracy, attribute binding, and spatial relationships. Furthermore, by simply adjusting the instructions to the LLM, SLD can perform image editing tasks, bridging the gap between text-to-image generation and image editing pipelines. We will make our code available for future research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.15813v1">FlowZero: Zero-Shot Text-to-Video Synthesis with LLM-Driven Dynamic Scene Syntax</a></div>
    <div class="paper-meta">
      📅 2023-11-27
      | 💬 Project page: https://flowzero-video.github.io
    </div>
    <details class="paper-abstract">
      Text-to-video (T2V) generation is a rapidly growing research area that aims to translate the scenes, objects, and actions within complex video text into a sequence of coherent visual frames. We present FlowZero, a novel framework that combines Large Language Models (LLMs) with image diffusion models to generate temporally-coherent videos. FlowZero uses LLMs to understand complex spatio-temporal dynamics from text, where LLMs can generate a comprehensive dynamic scene syntax (DSS) containing scene descriptions, object layouts, and background motion patterns. These elements in DSS are then used to guide the image diffusion model for video generation with smooth object motions and frame-to-frame coherence. Moreover, FlowZero incorporates an iterative self-refinement process, enhancing the alignment between the spatio-temporal layouts and the textual prompts for the videos. To enhance global coherence, we propose enriching the initial noise of each frame with motion dynamics to control the background movement and camera motion adaptively. By using spatio-temporal syntaxes to guide the diffusion process, FlowZero achieves improvement in zero-shot video synthesis, generating coherent videos with vivid motion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.15759v1">Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-27
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual-language understanding, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance overall capabilities of LLMs, which could be regraded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce the Modular Visual Memory, a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixtures-of-Multimodal Experts architecture in LLMs to invoke multimodal knowledge collaboration during generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on multimodal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.15698v1">Cerbero-7B: A Leap Forward in Language-Specific LLMs Through Enhanced Chat Corpus Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2023-11-27
    </div>
    <details class="paper-abstract">
      This study introduces a novel approach for generating high-quality, language-specific chat corpora using a self-chat mechanism. We combine a generator LLM for creating new samples and an embedder LLM to ensure diversity. A new Masked Language Modelling (MLM) model-based quality assessment metric is proposed for evaluating and filtering the corpora. Utilizing the llama2-70b as the generator and a multilingual sentence transformer as embedder, we generate an Italian chat corpus and refine the Fauno corpus, which is based on translated English ChatGPT self-chat data. The refinement uses structural assertions and Natural Language Processing techniques. Both corpora undergo a comprehensive quality evaluation using the proposed MLM model-based quality metric. The Italian LLM fine-tuned with these corpora demonstrates significantly enhanced language comprehension and question-answering skills. The resultant model, cerbero-7b, establishes a new state-of-the-art for Italian LLMs. This approach marks a substantial advancement in the development of language-specific LLMs, with a special emphasis on augmenting corpora for underrepresented languages like Italian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11853v3">How to Prompt LLMs for Text-to-SQL: A Study in Zero-shot, Single-domain, and Cross-domain Settings</a></div>
    <div class="paper-meta">
      📅 2023-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with in-context learning have demonstrated remarkable capability in the text-to-SQL task. Previous research has prompted LLMs with various demonstration-retrieval strategies and intermediate reasoning steps to enhance the performance of LLMs. However, those works often employ varied strategies when constructing the prompt text for text-to-SQL inputs, such as databases and demonstration examples. This leads to a lack of comparability in both the prompt constructions and their primary contributions. Furthermore, selecting an effective prompt construction has emerged as a persistent problem for future research. To address this limitation, we comprehensively investigate the impact of prompt constructions across various settings and provide insights into prompt constructions for future text-to-SQL studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.15759v1">Towards Vision Enhancing LLMs: Empowering Multimodal Knowledge Storage and Sharing in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-27
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in multimodal large language models (MLLMs) have achieved significant multimodal generation capabilities, akin to GPT-4. These models predominantly map visual information into language representation space, leveraging the vast knowledge and powerful text generation abilities of LLMs to produce multimodal instruction-following responses. We could term this method as LLMs for Vision because of its employing LLMs for visual-language understanding, yet observe that these MLLMs neglect the potential of harnessing visual knowledge to enhance overall capabilities of LLMs, which could be regraded as Vision Enhancing LLMs. In this paper, we propose an approach called MKS2, aimed at enhancing LLMs through empowering Multimodal Knowledge Storage and Sharing in LLMs. Specifically, we introduce the Modular Visual Memory, a component integrated into the internal blocks of LLMs, designed to store open-world visual information efficiently. Additionally, we present a soft Mixtures-of-Multimodal Experts architecture in LLMs to invoke multimodal knowledge collaboration during generation. Our comprehensive experiments demonstrate that MKS2 substantially augments the reasoning capabilities of LLMs in contexts necessitating physical or commonsense knowledge. It also delivers competitive results on multimodal benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.14904v1">LLM-Assisted Code Cleaning For Training Accurate Code Generators</a></div>
    <div class="paper-meta">
      📅 2023-11-25
    </div>
    <details class="paper-abstract">
      Natural language to code generation is an important application area of LLMs and has received wide attention from the community. The majority of relevant studies have exclusively concentrated on increasing the quantity and functional correctness of training sets while disregarding other stylistic elements of programs. More recently, data quality has garnered a lot of interest and multiple works have showcased its importance for improving performance. In this work, we investigate data quality for code and find that making the code more structured and readable leads to improved code generation performance of the system. We build a novel data-cleaning pipeline that uses these principles to transform existing programs by 1.) renaming variables, 2.) modularizing and decomposing complex code into smaller helper sub-functions, and 3.) inserting natural-language based plans via LLM based transformations. We evaluate our approach on two challenging algorithmic code generation benchmarks and find that fine-tuning CodeLLaMa-7B on our transformed modularized programs improves the performance by up to 30% compared to fine-tuning on the original dataset. Additionally, we demonstrate improved performance from using a smaller amount of higher-quality data, finding that a model fine-tuned on the entire original dataset is outperformed by a model trained on 15% of our cleaned dataset. Even in comparison to closed-source models, our models outperform the much larger AlphaCoder models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.14876v1">Exploiting Large Language Models (LLMs) through Deception Techniques and Persuasion Principles</a></div>
    <div class="paper-meta">
      📅 2023-11-24
      | 💬 10 pages, 16 tables, 5 figures, IEEE BigData 2023 (Workshops)
    </div>
    <details class="paper-abstract">
      With the recent advent of Large Language Models (LLMs), such as ChatGPT from OpenAI, BARD from Google, Llama2 from Meta, and Claude from Anthropic AI, gain widespread use, ensuring their security and robustness is critical. The widespread use of these language models heavily relies on their reliability and proper usage of this fascinating technology. It is crucial to thoroughly test these models to not only ensure its quality but also possible misuses of such models by potential adversaries for illegal activities such as hacking. This paper presents a novel study focusing on exploitation of such large language models against deceptive interactions. More specifically, the paper leverages widespread and borrows well-known techniques in deception theory to investigate whether these models are susceptible to deceitful interactions. This research aims not only to highlight these risks but also to pave the way for robust countermeasures that enhance the security and integrity of language models in the face of sophisticated social engineering tactics. Through systematic experiments and analysis, we assess their performance in these critical security domains. Our results demonstrate a significant finding in that these large language models are susceptible to deception and social engineering attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.14030v1">PrivateLoRA For Efficient Privacy Preserving LLM</a></div>
    <div class="paper-meta">
      📅 2023-11-23
    </div>
    <details class="paper-abstract">
      End users face a choice between privacy and efficiency in current Large Language Model (LLM) service paradigms. In cloud-based paradigms, users are forced to compromise data locality for generation quality and processing speed. Conversely, edge device paradigms maintain data locality but fail to deliver satisfactory performance. In this work, we propose a novel LLM service paradigm that distributes privacy-sensitive computation on edge devices and shared computation in the cloud. Only activations are transmitted between the central cloud and edge devices to ensure data locality. Our core innovation, PrivateLoRA, addresses the challenging communication overhead by exploiting the low rank of residual activations, achieving over 95% communication reduction. Consequently, PrivateLoRA effectively maintains data locality and is extremely resource efficient. Under standard 5G networks, PrivateLoRA achieves throughput over 300% of device-only solutions for 7B models and over 80% of an A100 GPU for 33B models. PrivateLoRA also provides tuning performance comparable to LoRA for advanced personalization. Our approach democratizes access to state-of-the-art generative AI for edge devices, paving the way for more tailored LLM experiences for the general public. To our knowledge, our proposed framework is the first efficient and privacy-preserving LLM solution in the literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11462v2">LLM aided semi-supervision for Extractive Dialog Summarization</a></div>
    <div class="paper-meta">
      📅 2023-11-23
      | 💬 to be published in EMNLP Findings
    </div>
    <details class="paper-abstract">
      Generating high-quality summaries for chat dialogs often requires large labeled datasets. We propose a method to efficiently use unlabeled data for extractive summarization of customer-agent dialogs. In our method, we frame summarization as a question-answering problem and use state-of-the-art large language models (LLMs) to generate pseudo-labels for a dialog. We then use these pseudo-labels to fine-tune a chat summarization model, effectively transferring knowledge from the large LLM into a smaller specialized model. We demonstrate our method on the \tweetsumm dataset, and show that using 10% of the original labelled data set we can achieve 65.9/57.0/61.0 ROUGE-1/-2/-L, whereas the current state-of-the-art trained on the entire training data set obtains 65.16/55.81/64.37 ROUGE-1/-2/-L. In other words, in the worst case (i.e., ROUGE-L) we still effectively retain 94.7% of the performance while using only 10% of the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06622v2">TrainerAgent: Customizable and Efficient Model Training through LLM-Powered Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2023-11-23
    </div>
    <details class="paper-abstract">
      Training AI models has always been challenging, especially when there is a need for custom models to provide personalized services. Algorithm engineers often face a lengthy process to iteratively develop models tailored to specific business requirements, making it even more difficult for non-experts. The quest for high-quality and efficient model development, along with the emergence of Large Language Model (LLM) Agents, has become a key focus in the industry. Leveraging the powerful analytical, planning, and decision-making capabilities of LLM, we propose a TrainerAgent system comprising a multi-agent framework including Task, Data, Model and Server agents. These agents analyze user-defined tasks, input data, and requirements (e.g., accuracy, speed), optimizing them comprehensively from both data and model perspectives to obtain satisfactory models, and finally deploy these models as online service. Experimental evaluations on classical discriminative and generative tasks in computer vision and natural language processing domains demonstrate that our system consistently produces models that meet the desired criteria. Furthermore, the system exhibits the ability to critically identify and reject unattainable tasks, such as fantastical scenarios or unethical requests, ensuring robustness and safety. This research presents a significant advancement in achieving desired models with increased efficiency and quality as compared to traditional model development, facilitated by the integration of LLM-powered analysis, decision-making, and execution capabilities, as well as the collaboration among four agents. We anticipate that our work will contribute to the advancement of research on TrainerAgent in both academic and industry communities, potentially establishing it as a new paradigm for model development in the field of AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13784v1">DaG LLM ver 1.0: Pioneering Instruction-Tuned Language Modeling for Korean NLP</a></div>
    <div class="paper-meta">
      📅 2023-11-23
    </div>
    <details class="paper-abstract">
      This paper presents the DaG LLM (David and Goliath Large Language Model), a language model specialized for Korean and fine-tuned through Instruction Tuning across 41 tasks within 13 distinct categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13565v1">Drilling Down into the Discourse Structure with LLMs for Long Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-11-22
      | 💬 Accepted to the Findings of EMNLP 2023
    </div>
    <details class="paper-abstract">
      We address the task of evidence retrieval for long document question answering, which involves locating relevant paragraphs within a document to answer a question. We aim to assess the applicability of large language models (LLMs) in the task of zero-shot long document evidence retrieval, owing to their unprecedented performance across various NLP tasks. However, currently the LLMs can consume limited context lengths as input, thus providing document chunks as inputs might overlook the global context while missing out on capturing the inter-segment dependencies. Moreover, directly feeding the large input sets can incur significant computational costs, particularly when processing the entire document (and potentially incurring monetary expenses with enterprise APIs like OpenAI's GPT variants). To address these challenges, we propose a suite of techniques that exploit the discourse structure commonly found in documents. By utilizing this structure, we create a condensed representation of the document, enabling a more comprehensive understanding and analysis of relationships between different parts. We retain $99.6\%$ of the best zero-shot approach's performance, while processing only $26\%$ of the total tokens used by the best approach in the information seeking evidence retrieval setup. We also show how our approach can be combined with \textit{self-ask} reasoning agent to achieve best zero-shot performance in complex multi-hop question answering, just $\approx 4\%$ short of zero-shot performance using gold evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.13381v1">Confidant: Customizing Transformer-based LLMs via Collaborative Edge Training</a></div>
    <div class="paper-meta">
      📅 2023-11-22
      | 💬 6 pages, 7 figures; Submitted to HotMobile 2024
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have demonstrated impressive capabilities in a variety of natural language processing (NLP) tasks. Nonetheless, it is challenging to deploy and fine-tune LLMs on mobile edge devices with limited computing, memory, and energy budgets. In this paper, we propose Confidant, a multi-backend collaborative training framework for customizing state-of-the-art LLMs on commodity mobile devices like smartphones. Confidant partitions an LLM into several sub-models so that each fits into a mobile device's memory. A pipeline parallel training mechanism is further developed to ensure fast and efficient distributed training. In addition, we propose a novel backend scheduler to allocate different attention heads to heterogeneous compute hardware, including mobile CPU and GPUs, to maximize the compute resource utilization on each edge device. Our preliminary experimental results show that Confidant achieves at most 45.3% memory reduction and 8.03x inference speedup in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00603v2">Faithful Explanations of Black-box NLP Models Using LLM-generated Counterfactuals</a></div>
    <div class="paper-meta">
      📅 2023-11-22
    </div>
    <details class="paper-abstract">
      Causal explanations of the predictions of NLP systems are essential to ensure safety and establish trust. Yet, existing methods often fall short of explaining model predictions effectively or efficiently and are often model-specific. In this paper, we address model-agnostic explanations, proposing two approaches for counterfactual (CF) approximation. The first approach is CF generation, where a large language model (LLM) is prompted to change a specific text concept while keeping confounding concepts unchanged. While this approach is demonstrated to be very effective, applying LLM at inference-time is costly. We hence present a second approach based on matching, and propose a method that is guided by an LLM at training-time and learns a dedicated embedding space. This space is faithful to a given causal graph and effectively serves to identify matches that approximate CFs. After showing theoretically that approximating CFs is required in order to construct faithful explanations, we benchmark our approaches and explain several models, including LLMs with billions of parameters. Our empirical results demonstrate the excellent performance of CF generation models as model-agnostic explainers. Moreover, our matching approach, which requires far less test-time resources, also provides effective explanations, surpassing many baselines. We also find that Top-K techniques universally improve every tested method. Finally, we showcase the potential of LLMs in constructing new benchmarks for model explanation and subsequently validate our conclusions. Our work illuminates new pathways for efficient and accurate approaches to interpreting NLP systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.12287v1">Adapting LLMs for Efficient, Personalized Information Retrieval: Methods and Implications</a></div>
    <div class="paper-meta">
      📅 2023-11-21
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) heralds a pivotal shift in online user interactions with information. Traditional Information Retrieval (IR) systems primarily relied on query-document matching, whereas LLMs excel in comprehending and generating human-like text, thereby enriching the IR experience significantly. While LLMs are often associated with chatbot functionalities, this paper extends the discussion to their explicit application in information retrieval. We explore methodologies to optimize the retrieval process, select optimal models, and effectively scale and orchestrate LLMs, aiming for cost-efficiency and enhanced result accuracy. A notable challenge, model hallucination-where the model yields inaccurate or misinterpreted data-is addressed alongside other model-specific hurdles. Our discourse extends to crucial considerations including user privacy, data optimization, and the necessity for system clarity and interpretability. Through a comprehensive examination, we unveil not only innovative strategies for integrating Language Models (LLMs) with Information Retrieval (IR) systems, but also the consequential considerations that underline the need for a balanced approach aligned with user-centric principles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11628v1">Incorporating LLM Priors into Tabular Learners</a></div>
    <div class="paper-meta">
      📅 2023-11-20
      | 💬 Table Representation Learning Workshop at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      We present a method to integrate Large Language Models (LLMs) and traditional tabular data classification techniques, addressing LLMs challenges like data serialization sensitivity and biases. We introduce two strategies utilizing LLMs for ranking categorical variables and generating priors on correlations between continuous variables and targets, enhancing performance in few-shot scenarios. We focus on Logistic Regression, introducing MonotonicLR that employs a non-linear monotonic function for mapping ordinals to cardinals while preserving LLM-determined orders. Validation against baseline models reveals the superior performance of our approach, especially in low-data scenarios, while remaining interpretable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11226v1">An Interactive Query Generation Assistant using LLM-based Prompt Modification and User Feedback</a></div>
    <div class="paper-meta">
      📅 2023-11-19
      | 💬 Intelligence Advanced Research Projects Activity (IARPA) BETTER Research Program
    </div>
    <details class="paper-abstract">
      While search is the predominant method of accessing information, formulating effective queries remains a challenging task, especially for situations where the users are not familiar with a domain, or searching for documents in other languages, or looking for complex information such as events, which are not easily expressible as queries. Providing example documents or passages of interest, might be easier for a user, however, such query-by-example scenarios are prone to concept drift, and are highly sensitive to the query generation method. This demo illustrates complementary approaches of using LLMs interactively, assisting and enabling the user to provide edits and feedback at all stages of the query formulation process. The proposed Query Generation Assistant is a novel search interface which supports automatic and interactive query generation over a mono-linguial or multi-lingual document collection. Specifically, the proposed assistive interface enables the users to refine the queries generated by different LLMs, to provide feedback on the retrieved documents or passages, and is able to incorporate the users' feedback as prompts to generate more effective queries. The proposed interface is a valuable experimental tool for exploring fine-tuning and prompting of LLMs for query generation to qualitatively evaluate the effectiveness of retrieval and ranking models, and for conducting Human-in-the-Loop (HITL) experiments for complex search tasks where users struggle to formulate queries without such assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.10401v1">LLM-based Control Code Generation using Image Recognition</a></div>
    <div class="paper-meta">
      📅 2023-11-17
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      LLM-based code generation could save significant manual efforts in industrial automation, where control engineers manually produce control logic for sophisticated production processes. Previous attempts in control logic code generation lacked methods to interpret schematic drawings from process engineers. Recent LLMs now combine image recognition, trained domain knowledge, and coding skills. We propose a novel LLM-based code generation method that generates IEC 61131-3 Structure Text control logic source code from Piping-and-Instrumentation Diagrams (P&IDs) using image recognition. We have evaluated the method in three case study with industrial P&IDs and provide first evidence on the feasibility of such a code generation besides experiences on image recognition glitches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.10215v1">Predictive Minds: LLMs As Atypical Active Inference Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-16
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like GPT are often conceptualized as passive predictors, simulators, or even stochastic parrots. We instead conceptualize LLMs by drawing on the theory of active inference originating in cognitive science and neuroscience. We examine similarities and differences between traditional active inference systems and LLMs, leading to the conclusion that, currently, LLMs lack a tight feedback loop between acting in the world and perceiving the impacts of their actions, but otherwise fit in the active inference paradigm. We list reasons why this loop may soon be closed, and possible consequences of this including enhanced model self-awareness and the drive to minimize prediction error by changing the world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.11860v2">Let's Sample Step by Step: Adaptive-Consistency for Efficient Reasoning and Coding with LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-16
      | 💬 Published at EMNLP 2023
    </div>
    <details class="paper-abstract">
      A popular approach for improving the correctness of output from large language models (LLMs) is Self-Consistency - poll the LLM multiple times and output the most frequent solution. Existing Self-Consistency techniques always generate a constant number of samples per question, where a better approach will be to non-uniformly distribute the available budget based on the amount of agreement in the samples generated so far. In response, we introduce Adaptive-Consistency, a cost-efficient, model-agnostic technique that dynamically adjusts the number of samples per question using a lightweight stopping criterion. Our experiments over 17 reasoning and code generation datasets and three LLMs demonstrate that Adaptive-Consistency reduces sample budget by up to 7.9 times with an average accuracy drop of less than 0.1%. Our code and data are available at https://www.sample-step-by-step.info
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09841v1">Leveraging LLMs in Scholarly Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-11-16
    </div>
    <details class="paper-abstract">
      This paper presents a scholarly Knowledge Graph Question Answering (KGQA) that answers bibliographic natural language questions by leveraging a large language model (LLM) in a few-shot manner. The model initially identifies the top-n similar training questions related to a given test question via a BERT-based sentence encoder and retrieves their corresponding SPARQL. Using the top-n similar question-SPARQL pairs as an example and the test question creates a prompt. Then pass the prompt to the LLM and generate a SPARQL. Finally, runs the SPARQL against the underlying KG - ORKG (Open Research KG) endpoint and returns an answer. Our system achieves an F1 score of 99.0%, on SciQA - one of the Scholarly-QALD-23 challenge benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09825v1">Human Still Wins over LLM: An Empirical Study of Active Learning on Domain-Specific Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2023-11-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated considerable advances, and several claims have been made about their exceeding human performance. However, in real-world tasks, domain knowledge is often required. Low-resource learning methods like Active Learning (AL) have been proposed to tackle the cost of domain expert annotation, raising this question: Can LLMs surpass compact models trained with expert annotations in domain-specific tasks? In this work, we conduct an empirical experiment on four datasets from three different domains comparing SOTA LLMs with small models trained on expert annotations with AL. We found that small models can outperform GPT-3.5 with a few hundreds of labeled data, and they achieve higher or similar performance with GPT-4 despite that they are hundreds time smaller. Based on these findings, we posit that LLM predictions can be used as a warmup method in real-world applications and human experts remain indispensable in tasks involving data annotation driven by domain-specific knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09721v1">On Evaluating the Integration of Reasoning and Action in LLM Agents with Database Question Answering</a></div>
    <div class="paper-meta">
      📅 2023-11-16
    </div>
    <details class="paper-abstract">
      This study introduces a new long-form database question answering dataset designed to evaluate how Large Language Models (LLMs) interact with a SQL interpreter. The task necessitates LLMs to strategically generate multiple SQL queries to retrieve sufficient data from a database, to reason with the acquired context, and to synthesize them into a comprehensive analytical narrative. Our findings highlight that this task poses great challenges even for the state-of-the-art GPT-4 model. We propose and evaluate two interaction strategies, and provide a fine-grained analysis of the individual stages within the interaction. A key discovery is the identification of two primary bottlenecks hindering effective interaction: the capacity for planning and the ability to generate multiple SQL queries. To address the challenge of accurately assessing answer quality, we introduce a multi-agent evaluation framework that simulates the academic peer-review process, enhancing the precision and reliability of our evaluations. This framework allows for a more nuanced understanding of the strengths and limitations of current LLMs in complex retrieval and reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09550v1">A Speed Odyssey for Deployable Quantization of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-16
    </div>
    <details class="paper-abstract">
      The large language model era urges faster and less costly inference. Prior model compression works on LLMs tend to undertake a software-centric approach primarily focused on the simulated quantization performance. By neglecting the feasibility of deployment, these approaches are typically disabled in real practice. They used to drastically push down the quantization bit range for a reduced computation which might not be supported by the mainstream hardware, or involve sophisticated algorithms that introduce extra computation or memory access overhead. We argue that pursuing a hardware-centric approach in the construction of quantization algorithms is crucial. In this regard, we are driven to build our compression method on top of hardware awareness, eliminating impractical algorithm choices while maximizing the benefit of hardware acceleration. Our method, OdysseyLLM, comes with a novel W4A8 kernel implementation called FastGEMM and a combined recipe of quantization strategies. Extensive experiments manifest the superiority of our W4A8 method which brings the actual speed boosting up to \textbf{4$\times$} compared to Hugging Face FP16 inference and \textbf{2.23$\times$} vs. the state-of-the-art inference engine TensorRT-LLM in FP16, and \textbf{1.45$\times$} vs. TensorRT-LLM in INT8, yet without substantially harming the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.14711v1">Towards Publicly Accountable Frontier LLMs: Building an External Scrutiny Ecosystem under the ASPIRE Framework</a></div>
    <div class="paper-meta">
      📅 2023-11-15
      | 💬 Accepted to Workshop on Socially Responsible Language Modelling Research (SoLaR) at the 2023 Conference on Neural Information Processing Systems (NeurIPS 2023)
    </div>
    <details class="paper-abstract">
      With the increasing integration of frontier large language models (LLMs) into society and the economy, decisions related to their training, deployment, and use have far-reaching implications. These decisions should not be left solely in the hands of frontier LLM developers. LLM users, civil society and policymakers need trustworthy sources of information to steer such decisions for the better. Involving outside actors in the evaluation of these systems - what we term 'external scrutiny' - via red-teaming, auditing, and external researcher access, offers a solution. Though there are encouraging signs of increasing external scrutiny of frontier LLMs, its success is not assured. In this paper, we survey six requirements for effective external scrutiny of frontier AI systems and organize them under the ASPIRE framework: Access, Searching attitude, Proportionality to the risks, Independence, Resources, and Expertise. We then illustrate how external scrutiny might function throughout the AI lifecycle and offer recommendations to policymakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08723v1">Token Prediction as Implicit Classification to Identify LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2023-11-15
      | 💬 EMNLP 2023, Main Conference
    </div>
    <details class="paper-abstract">
      This paper introduces a novel approach for identifying the possible large language models (LLMs) involved in text generation. Instead of adding an additional classification layer to a base LM, we reframe the classification task as a next-token prediction task and directly fine-tune the base LM to perform it. We utilize the Text-to-Text Transfer Transformer (T5) model as the backbone for our experiments. We compared our approach to the more direct approach of utilizing hidden states for classification. Evaluation shows the exceptional performance of our method in the text classification task, highlighting its simplicity and efficiency. Furthermore, interpretability studies on the features extracted by our model reveal its ability to differentiate distinctive writing styles among various LLMs even in the absence of an explicit classifier. We also collected a dataset named OpenLLMText, containing approximately 340k text samples from human and LLMs, including GPT3.5, PaLM, LLaMA, and GPT2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08719v1">Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory</a></div>
    <div class="paper-meta">
      📅 2023-11-15
    </div>
    <details class="paper-abstract">
      Memory-augmented Large Language Models (LLMs) have demonstrated remarkable performance in long-term human-machine interactions, which basically relies on iterative recalling and reasoning of history to generate high-quality responses. However, such repeated recall-reason steps easily produce biased thoughts, \textit{i.e.}, inconsistent reasoning results when recalling the same history for different questions. On the contrary, humans can keep thoughts in the memory and recall them without repeated reasoning. Motivated by this human capability, we propose a novel memory mechanism called TiM (Think-in-Memory) that enables LLMs to maintain an evolved memory for storing historical thoughts along the conversation stream. The TiM framework consists of two crucial stages: (1) before generating a response, a LLM agent recalls relevant thoughts from memory, and (2) after generating a response, the LLM agent post-thinks and incorporates both historical and new thoughts to update the memory. Thus, TiM can eliminate the issue of repeated reasoning by saving the post-thinking thoughts as the history. Besides, we formulate the basic principles to organize the thoughts in memory based on the well-established operations, (\textit{i.e.}, insert, forget, and merge operations), allowing for dynamic updates and evolution of the thoughts. Furthermore, we introduce Locality-Sensitive Hashing into TiM to achieve efficient retrieval for the long-term conversations. We conduct qualitative and quantitative experiments on real-world and simulated dialogues covering a wide range of topics, demonstrating that equipping existing LLMs with TiM significantly enhances their performance in generating responses for long-term interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.12342v2">Eliminating Reasoning via Inferring with Planning: A New Framework to Guide LLMs' Non-linear Thinking</a></div>
    <div class="paper-meta">
      📅 2023-11-15
    </div>
    <details class="paper-abstract">
      Chain-of-Thought(CoT) prompting and its variants explore equipping large language models (LLMs) with high-level reasoning abilities by emulating human-like linear cognition and logic. However, the human mind is complicated and mixed with both linear and nonlinear thinking. In this work, we propose \textbf{I}nferential \textbf{E}xclusion \textbf{P}rompting (IEP), a novel prompting that combines the principles of elimination and inference in order to guide LLMs to think non-linearly. IEP guides LLMs to plan and then utilize Natural Language Inference (NLI) to deduce each possible solution's entailment relation with context, commonsense, or facts, therefore yielding a broader perspective by thinking back for inferring. This forward planning and backward eliminating process allows IEP to better simulate the complex human thinking processes compared to other CoT-based methods, which only reflect linear cognitive processes. We conducted a series of empirical studies and have corroborated that IEP consistently outperforms CoT across various tasks. Additionally, we observe that integrating IEP and CoT further improves the LLMs' performance on certain tasks, highlighting the necessity of equipping LLMs with mixed logic processes. Moreover, to better evaluate comprehensive features inherent in human logic, we introduce \textbf{M}ental-\textbf{A}bility \textbf{R}easoning \textbf{B}enchmark (MARB). The benchmark comprises six novel subtasks with a total of 9,115 questions, among which 1,685 are developed with hand-crafted rationale references. We believe both \textsc{IEP} and \textsc{MARB} can serve as a promising direction for unveiling LLMs' logic and verbal reasoning abilities and drive further advancements. \textsc{MARB} will be available at ~\texttt{anonymity link} soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08147v1">RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge</a></div>
    <div class="paper-meta">
      📅 2023-11-14
    </div>
    <details class="paper-abstract">
      LLMs and AI chatbots have improved people's efficiency in various fields. However, the necessary knowledge for answering the question may be beyond the models' knowledge boundaries. To mitigate this issue, many researchers try to introduce external knowledge, such as knowledge graphs and Internet contents, into LLMs for up-to-date information. However, the external information from the Internet may include counterfactual information that will confuse the model and lead to an incorrect response. Thus there is a pressing need for LLMs to possess the ability to distinguish reliable information from external knowledge. Therefore, to evaluate the ability of LLMs to discern the reliability of external knowledge, we create a benchmark from existing knowledge bases. Our benchmark consists of two tasks, Question Answering and Text Generation, and for each task, we provide models with a context containing counterfactual information. Evaluation results show that existing LLMs are susceptible to interference from unreliable external knowledge with counterfactual information, and simple intervention methods make limited contributions to the alleviation of this issue.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08117v1">Insights into Classifying and Mitigating LLMs' Hallucinations</a></div>
    <div class="paper-meta">
      📅 2023-11-14
      | 💬 Accepted at AIxIA 2023
    </div>
    <details class="paper-abstract">
      The widespread adoption of large language models (LLMs) across diverse AI applications is proof of the outstanding achievements obtained in several tasks, such as text mining, text generation, and question answering. However, LLMs are not exempt from drawbacks. One of the most concerning aspects regards the emerging problematic phenomena known as "Hallucinations". They manifest in text generation systems, particularly in question-answering systems reliant on LLMs, potentially resulting in false or misleading information propagation. This paper delves into the underlying causes of AI hallucination and elucidates its significance in artificial intelligence. In particular, Hallucination classification is tackled over several tasks (Machine Translation, Question and Answer, Dialog Systems, Summarisation Systems, Knowledge Graph with LLMs, and Visual Question Answer). Additionally, we explore potential strategies to mitigate hallucinations, aiming to enhance the overall reliability of LLMs. Our research addresses this critical issue within the HeReFaNMi (Health-Related Fake News Mitigation) project, generously supported by NGI Search, dedicated to combating Health-Related Fake News dissemination on the Internet. This endeavour represents a concerted effort to safeguard the integrity of information dissemination in an age of evolving AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07961v1">The ART of LLM Refinement: Ask, Refine, and Trust</a></div>
    <div class="paper-meta">
      📅 2023-11-14
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated remarkable generative abilities, but can they judge the quality of their own generations? A popular concept, referred to as self-refinement, postulates that LLMs can detect and correct the errors in their generations when asked to do so. However, recent empirical evidence points in the opposite direction, suggesting that LLMs often struggle to accurately identify errors when reasoning is involved. To address this, we propose a reasoning with refinement objective called ART: Ask, Refine, and Trust, which asks necessary questions to decide when an LLM should refine its output, and either affirm or withhold trust in its refinement by ranking the refinement and the initial prediction. On two multistep reasoning tasks of mathematical word problems (GSM8K) and question answering (StrategyQA), ART achieves a performance gain of +5 points over self-refinement baselines, while using a much smaller model as the decision maker. We also demonstrate the benefit of using smaller models to make refinement decisions as a cost-effective alternative to fine-tuning a larger model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.19347v3">Improving Factual Consistency of Text Summarization by Adversarially Decoupling Comprehension and Embellishment Abilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-14
    </div>
    <details class="paper-abstract">
      Despite the recent progress in text summarization made by large language models (LLMs), they often generate summaries that are factually inconsistent with original articles, known as "hallucinations" in text generation. Unlike previous small models (e.g., BART, T5), current LLMs make fewer silly mistakes but more sophisticated ones, such as imposing cause and effect, adding false details, overgeneralizing, etc. These hallucinations are challenging to detect through traditional methods, which poses great challenges for improving the factual consistency of text summarization. In this paper, we propose an adversarially DEcoupling method to disentangle the Comprehension and EmbellishmeNT abilities of LLMs (DECENT). Furthermore, we adopt a probing-based efficient training to cover the shortage of sensitivity for true and false in the training process of LLMs. In this way, LLMs are less confused about embellishing and understanding; thus, they can execute the instructions more accurately and have enhanced abilities to distinguish hallucinations. Experimental results show that DECENT significantly improves the reliability of text summarization based on LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07897v1">CPopQA: Ranking Cultural Concept Popularity by LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-14
    </div>
    <details class="paper-abstract">
      Prior work has demonstrated large language models' (LLMs) potential to discern statistical tendencies within their pre-training corpora. Despite that, many examinations of LLMs' knowledge capacity focus on knowledge explicitly appearing in the training data or implicitly inferable from similar contexts. How well an LLM captures the corpus-level statistical trends of concepts for reasoning, especially long-tail ones, is still underexplored. In this study, we introduce a novel few-shot question-answering task (CPopQA) that examines LLMs' statistical ranking abilities for long-tail cultural concepts (e.g., holidays), with a specific focus on these concepts' popularity in the United States and the United Kingdom, respectively. We curate a dataset containing 459 holidays across 58 countries, generating a total of 6,000 QA testing pairs. Experiments on four strong LLMs show that large models are capable of ranking long-tail cultural concepts regarding their statistical tendency. Notably, GPT-3.5 displayed superior performance and exhibited its potential to identify geo-cultural proximity across continents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07689v1">MART: Improving LLM Safety with Multi-round Automatic Red-Teaming</a></div>
    <div class="paper-meta">
      📅 2023-11-13
    </div>
    <details class="paper-abstract">
      Red-teaming is a common practice for mitigating unsafe behaviors in Large Language Models (LLMs), which involves thoroughly assessing LLMs to identify potential flaws and addressing them with responsible and accurate responses. While effective, manual red-teaming is costly, and existing automatic red-teaming typically discovers safety risks without addressing them. In this paper, we propose a Multi-round Automatic Red-Teaming (MART) method, which incorporates both automatic adversarial prompt writing and safe response generation, significantly increasing red-teaming scalability and the safety of the target LLM. Specifically, an adversarial LLM and a target LLM interplay with each other in an iterative manner, where the adversarial LLM aims to generate challenging prompts that elicit unsafe responses from the target LLM, while the target LLM is fine-tuned with safety aligned data on these adversarial prompts. In each round, the adversarial LLM crafts better attacks on the updated target LLM, while the target LLM also improves itself through safety fine-tuning. On adversarial prompt benchmarks, the violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing. Notably, model helpfulness on non-adversarial prompts remains stable throughout iterations, indicating the target LLM maintains strong performance on instruction following.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11816v2">Learning to Generate Better Than Your LLM</a></div>
    <div class="paper-meta">
      📅 2023-11-13
      | 💬 23 pages, 5 figures, 7 tables, 4 algorithms
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful paradigm for fine-tuning Large Language Models (LLMs) for text generation. In particular, recent LLMs such as ChatGPT and GPT-4 can engage in fluent conversations with users after finetuning with RL. Capitalizing on key properties of text generation, we seek to investigate RL algorithms beyond general purpose algorithms like Proximal Policy Optimization (PPO). In particular, we extend RL algorithms to allow them to interact with a dynamic black-box guide LLM and propose RL with guided feedback (RLGF), a suite of RL algorithms for LLM fine-tuning. We provide two ways for the guide LLM to interact with the LLM to be optimized for maximizing rewards. The guide LLM can generate text which serves as additional starting states for the RL optimization procedure. The guide LLM can also be used to complete the partial sentences generated by the LLM that is being optimized, treating the guide LLM as an expert to imitate and surpass eventually. We experiment on the IMDB positive sentiment, CommonGen, and TL;DR summarization tasks. We show that our RL algorithms achieve higher performance than supervised learning (SL) and the RL baseline PPO, demonstrating the benefit of interaction with the guide LLM. On both CommonGen and TL;DR, we not only outperform our SL baselines but also improve upon PPO across a variety of metrics beyond the one we optimized for. Our code can be found at https://github.com/Cornell-RL/tril.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05163v3">An Investigation of LLMs' Inefficacy in Understanding Converse Relations</a></div>
    <div class="paper-meta">
      📅 2023-11-13
      | 💬 Accepted by EMNLP 2023
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in many formal language oriented tasks, such as structural data-to-text and semantic parsing. However current benchmarks mostly follow the data distribution of the pre-training data of LLMs. Therefore, a natural question rises that do LLMs really understand the structured semantics of formal languages. In this paper, we investigate this problem on a special case, converse binary relation. We introduce a new benchmark ConvRe focusing on converse relations, which contains 17 relations and 1240 triples extracted from popular knowledge graph completion datasets. Our ConvRE features two tasks, Re2Text and Text2Re, which are formulated as multi-choice question answering to evaluate LLMs' ability to determine the matching between relations and associated text. For the evaluation protocol, apart from different prompting methods, we further introduce variants to the test text and few-shot example text. We conduct experiments on three popular LLM families and have observed various scaling trends. The results suggest that LLMs often resort to shortcut learning and still face challenges on our proposed benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.00477v2">Demonstration of InsightPilot: An LLM-Empowered Automated Data Exploration System</a></div>
    <div class="paper-meta">
      📅 2023-11-13
    </div>
    <details class="paper-abstract">
      Exploring data is crucial in data analysis, as it helps users understand and interpret the data more effectively. However, performing effective data exploration requires in-depth knowledge of the dataset and expertise in data analysis techniques. Not being familiar with either can create obstacles that make the process time-consuming and overwhelming for data analysts. To address this issue, we introduce InsightPilot, an LLM (Large Language Model)-based, automated data exploration system designed to simplify the data exploration process. InsightPilot automatically selects appropriate analysis intents, such as understanding, summarizing, and explaining. Then, these analysis intents are concretized by issuing corresponding intentional queries (IQueries) to create a meaningful and coherent exploration sequence. In brief, an IQuery is an abstraction and automation of data analysis operations, which mimics the approach of data analysts and simplifies the exploration process for users. By employing an LLM to iteratively collaborate with a state-of-the-art insight engine via IQueries, InsightPilot is effective in analyzing real-world datasets, enabling users to gain valuable insights through natural language inquiries. We demonstrate the effectiveness of InsightPilot in a case study, showing how it can help users gain valuable insights from their datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.05656v1">Combating Misinformation in the Age of LLMs: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2023-11-13
      | 💬 9 pages for the main paper, 35 pages including 656 references, more resources on "LLMs Meet Misinformation" are on the website: https://llm-misinformation.github.io/
    </div>
    <details class="paper-abstract">
      Misinformation such as fake news and rumors is a serious threat on information ecosystems and public trust. The emergence of Large Language Models (LLMs) has great potential to reshape the landscape of combating misinformation. Generally, LLMs can be a double-edged sword in the fight. On the one hand, LLMs bring promising opportunities for combating misinformation due to their profound world knowledge and strong reasoning abilities. Thus, one emergent question is: how to utilize LLMs to combat misinformation? On the other hand, the critical challenge is that LLMs can be easily leveraged to generate deceptive misinformation at scale. Then, another important question is: how to combat LLM-generated misinformation? In this paper, we first systematically review the history of combating misinformation before the advent of LLMs. Then we illustrate the current efforts and present an outlook for these two fundamental questions respectively. The goal of this survey paper is to facilitate the progress of utilizing LLMs for fighting misinformation and call for interdisciplinary efforts from different stakeholders for combating LLM-generated misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06772v1">ChatAnything: Facetime Chat with LLM-Enhanced Personas</a></div>
    <div class="paper-meta">
      📅 2023-11-12
    </div>
    <details class="paper-abstract">
      In this technical report, we target generating anthropomorphized personas for LLM-based characters in an online manner, including visual appearance, personality and tones, with only text descriptions. To achieve this, we first leverage the in-context learning capability of LLMs for personality generation by carefully designing a set of system prompts. We then propose two novel concepts: the mixture of voices (MoV) and the mixture of diffusers (MoD) for diverse voice and appearance generation. For MoV, we utilize the text-to-speech (TTS) algorithms with a variety of pre-defined tones and select the most matching one based on the user-provided text description automatically. For MoD, we combine the recent popular text-to-image generation techniques and talking head algorithms to streamline the process of generating talking objects. We termed the whole framework as ChatAnything. With it, users could be able to animate anything with any personas that are anthropomorphic using just a few text inputs. However, we have observed that the anthropomorphic objects produced by current generative models are often undetectable by pre-trained face landmark detectors, leading to failure of the face motion generation, even if these faces possess human-like appearances because those images are nearly seen during the training (e.g., OOD samples). To address this issue, we incorporate pixel-level guidance to infuse human face landmarks during the image generation phase. To benchmark these metrics, we have built an evaluation dataset. Based on it, we verify that the detection rate of the face landmark is significantly increased from 57.0% to 92.5% thus allowing automatic face animation based on generated speech content. The code and more results can be found at https://chatanything.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09243v1">Evaluating the Efficacy of Interactive Language Therapy Based on LLM for High-Functioning Autistic Adolescent Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2023-11-12
    </div>
    <details class="paper-abstract">
      This study investigates the efficacy of Large Language Models (LLMs) in interactive language therapy for high-functioning autistic adolescents. With the rapid advancement of artificial intelligence, particularly in natural language processing, LLMs present a novel opportunity to augment traditional psychological counseling methods. This research primarily focuses on evaluating the LLM's ability to engage in empathetic, adaptable, and contextually appropriate interactions within a therapeutic setting. A comprehensive evaluation was conducted by a panel of clinical psychologists and psychiatrists using a specially developed scorecard. The assessment covered various aspects of the LLM's performance, including empathy, communication skills, adaptability, engagement, and the ability to establish a therapeutic alliance. The study avoided direct testing with patients, prioritizing privacy and ethical considerations, and instead relied on simulated scenarios to gauge the LLM's effectiveness. The results indicate that LLMs hold significant promise as supportive tools in therapy, demonstrating strengths in empathetic engagement and adaptability in conversation. However, challenges in achieving the depth of personalization and emotional understanding characteristic of human therapists were noted. The study also highlights the importance of ethical considerations in the application of AI in therapeutic contexts. This research provides valuable insights into the potential and limitations of using LLMs in psychological counseling for autistic adolescents. It lays the groundwork for future explorations into AI's role in mental health care, emphasizing the need for ongoing development to enhance the capabilities of these models in therapeutic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.11689v2">Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-11
      | 💬 Paper published at Findings of the Association for Computational Linguistics: EMNLP, 2023
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently shown great advances in a variety of tasks, including natural language understanding and generation. However, their use in high-stakes decision-making scenarios is still limited due to the potential for errors. Selective prediction is a technique that can be used to improve the reliability of the LLMs by allowing them to abstain from making predictions when they are unsure of the answer. In this work, we propose a novel framework for adaptation with self-evaluation to improve the selective prediction performance of LLMs. Our framework is based on the idea of using parameter-efficient tuning to adapt the LLM to the specific task at hand while improving its ability to perform self-evaluation. We evaluate our method on a variety of question-answering (QA) datasets and show that it outperforms state-of-the-art selective prediction methods. For example, on the CoQA benchmark, our method improves the AUACC from 91.23% to 92.63% and improves the AUROC from 74.61% to 80.25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06612v1">PerceptionGPT: Effectively Fusing Visual Perception into LLM</a></div>
    <div class="paper-meta">
      📅 2023-11-11
    </div>
    <details class="paper-abstract">
      The integration of visual inputs with large language models (LLMs) has led to remarkable advancements in multi-modal capabilities, giving rise to visual large language models (VLLMs). However, effectively harnessing VLLMs for intricate visual perception tasks remains a challenge. In this paper, we present a novel end-to-end framework named PerceptionGPT, which efficiently and effectively equips the VLLMs with visual perception abilities by leveraging the representation power of LLMs' token embedding. Our proposed method treats the token embedding of the LLM as the carrier of spatial information, then leverage lightweight visual task encoders and decoders to perform visual perception tasks (e.g., detection, segmentation). Our approach significantly alleviates the training difficulty suffered by previous approaches that formulate the visual outputs as discrete tokens, and enables achieving superior performance with fewer trainable parameters, less training data and shorted training time. Moreover, as only one token embedding is required to decode the visual outputs, the resulting sequence length during inference is significantly reduced. Consequently, our approach enables accurate and flexible representations, seamless integration of visual perception tasks, and efficient handling of a multiple of visual outputs. We validate the effectiveness and efficiency of our approach through extensive experiments. The results demonstrate significant improvements over previous methods with much fewer trainable parameters and GPU hours, which facilitates future research in enabling LLMs with visual perception abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.07599v1">Testing LLMs on Code Generation with Varying Levels of Prompt Specificity</a></div>
    <div class="paper-meta">
      📅 2023-11-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated unparalleled prowess in mimicking human-like text generation and processing. Among the myriad of applications that benefit from LLMs, automated code generation is increasingly promising. The potential to transform natural language prompts into executable code promises a major shift in software development practices and paves the way for significant reductions in manual coding efforts and the likelihood of human-induced errors. This paper reports the results of a study that evaluates the performance of various LLMs, such as Bard, ChatGPT-3.5, ChatGPT-4, and Claude-2, in generating Python for coding problems. We focus on how levels of prompt specificity impact the accuracy, time efficiency, and space efficiency of the generated code. A benchmark of 104 coding problems, each with four types of prompts with varying degrees of tests and specificity, was employed to examine these aspects comprehensively. Our results indicate significant variations in performance across different LLMs and prompt types, and its key contribution is to reveal the ideal prompting strategy for creating accurate Python functions. This study lays the groundwork for further research in LLM capabilities and suggests practical implications for utilizing LLMs in automated code generation tasks and test-driven development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06102v1">Making LLMs Worth Every Penny: Resource-Limited Text Classification in Banking</a></div>
    <div class="paper-meta">
      📅 2023-11-10
      | 💬 Long paper accepted to ACM ICAIF-23
    </div>
    <details class="paper-abstract">
      Standard Full-Data classifiers in NLP demand thousands of labeled examples, which is impractical in data-limited domains. Few-shot methods offer an alternative, utilizing contrastive learning techniques that can be effective with as little as 20 examples per class. Similarly, Large Language Models (LLMs) like GPT-4 can perform effectively with just 1-5 examples per class. However, the performance-cost trade-offs of these methods remain underexplored, a critical concern for budget-limited organizations. Our work addresses this gap by studying the aforementioned approaches over the Banking77 financial intent detection dataset, including the evaluation of cutting-edge LLMs by OpenAI, Cohere, and Anthropic in a comprehensive set of few-shot scenarios. We complete the picture with two additional methods: first, a cost-effective querying method for LLMs based on retrieval-augmented generation (RAG), able to reduce operational costs multiple times compared to classic few-shot approaches, and second, a data augmentation method using GPT-4, able to improve performance in data-limited scenarios. Finally, to inspire future research, we provide a human expert's curated subset of Banking77, along with extensive error analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2311.05596v1">LLM Augmented Hierarchical Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-10
    </div>
    <details class="paper-abstract">
      Solving long-horizon, temporally-extended tasks using Reinforcement Learning (RL) is challenging, compounded by the common practice of learning without prior knowledge (or tabula rasa learning). Humans can generate and execute plans with temporally-extended actions and quickly learn to perform new tasks because we almost never solve problems from scratch. We want autonomous agents to have this same ability. Recently, LLMs have been shown to encode a tremendous amount of knowledge about the world and to perform impressive in-context learning and reasoning. However, using LLMs to solve real world problems is hard because they are not grounded in the current task. In this paper we exploit the planning capabilities of LLMs while using RL to provide learning from the environment, resulting in a hierarchical agent that uses LLMs to solve long-horizon tasks. Instead of completely relying on LLMs, they guide a high-level policy, making learning significantly more sample efficient. This approach is evaluated in simulation environments such as MiniGrid, SkillHack, and Crafter, and on a real robot arm in block manipulation tasks. We show that agents trained using our approach outperform other baselines methods and, once trained, don't need access to LLMs during deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05596v1">LLM Augmented Hierarchical Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-09
    </div>
    <details class="paper-abstract">
      Solving long-horizon, temporally-extended tasks using Reinforcement Learning (RL) is challenging, compounded by the common practice of learning without prior knowledge (or tabula rasa learning). Humans can generate and execute plans with temporally-extended actions and quickly learn to perform new tasks because we almost never solve problems from scratch. We want autonomous agents to have this same ability. Recently, LLMs have been shown to encode a tremendous amount of knowledge about the world and to perform impressive in-context learning and reasoning. However, using LLMs to solve real world problems is hard because they are not grounded in the current task. In this paper we exploit the planning capabilities of LLMs while using RL to provide learning from the environment, resulting in a hierarchical agent that uses LLMs to solve long-horizon tasks. Instead of completely relying on LLMs, they guide a high-level policy, making learning significantly more sample efficient. This approach is evaluated in simulation environments such as MiniGrid, SkillHack, and Crafter, and on a real robot arm in block manipulation tasks. We show that agents trained using our approach outperform other baselines methods and, once trained, don't need access to LLMs during deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05374v1">TencentLLMEval: A Hierarchical Evaluation of Real-World Capabilities for Human-Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown impressive capabilities across various natural language tasks. However, evaluating their alignment with human preferences remains a challenge. To this end, we propose a comprehensive human evaluation framework to assess LLMs' proficiency in following instructions on diverse real-world tasks. We construct a hierarchical task tree encompassing 7 major areas covering over 200 categories and over 800 tasks, which covers diverse capabilities such as question answering, reasoning, multiturn dialogue, and text generation, to evaluate LLMs in a comprehensive and in-depth manner. We also design detailed evaluation standards and processes to facilitate consistent, unbiased judgments from human evaluators. A test set of over 3,000 instances is released, spanning different difficulty levels and knowledge domains. Our work provides a standardized methodology to evaluate human alignment in LLMs for both English and Chinese. We also analyze the feasibility of automating parts of evaluation with a strong LLM (GPT-4). Our framework supports a thorough assessment of LLMs as they are integrated into real-world applications. We have made publicly available the task tree, TencentLLMEval dataset, and evaluation methodology which have been demonstrated as effective in assessing the performance of Tencent Hunyuan LLMs. By doing so, we aim to facilitate the benchmarking of advances in the development of safe and human-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.05656v1">Combating Misinformation in the Age of LLMs: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2023-11-09
      | 💬 9 pages for the main paper, 35 pages including 656 references, more resources on "LLMs Meet Misinformation" are on the website: https://llm-misinformation.github.io/
    </div>
    <details class="paper-abstract">
      Misinformation such as fake news and rumors is a serious threat on information ecosystems and public trust. The emergence of Large Language Models (LLMs) has great potential to reshape the landscape of combating misinformation. Generally, LLMs can be a double-edged sword in the fight. On the one hand, LLMs bring promising opportunities for combating misinformation due to their profound world knowledge and strong reasoning abilities. Thus, one emergent question is: how to utilize LLMs to combat misinformation? On the other hand, the critical challenge is that LLMs can be easily leveraged to generate deceptive misinformation at scale. Then, another important question is: how to combat LLM-generated misinformation? In this paper, we first systematically review the history of combating misinformation before the advent of LLMs. Then we illustrate the current efforts and present an outlook for these two fundamental questions respectively. The goal of this survey paper is to facilitate the progress of utilizing LLMs for fighting misinformation and call for interdisciplinary efforts from different stakeholders for combating LLM-generated misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.05656v1">Combating Misinformation in the Age of LLMs: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2023-11-09
      | 💬 9 pages for the main paper, 35 pages including 656 references, more resources on "LLMs Meet Misinformation" are on the website: https://llm-misinformation.github.io/
    </div>
    <details class="paper-abstract">
      Misinformation such as fake news and rumors is a serious threat on information ecosystems and public trust. The emergence of Large Language Models (LLMs) has great potential to reshape the landscape of combating misinformation. Generally, LLMs can be a double-edged sword in the fight. On the one hand, LLMs bring promising opportunities for combating misinformation due to their profound world knowledge and strong reasoning abilities. Thus, one emergent question is: how to utilize LLMs to combat misinformation? On the other hand, the critical challenge is that LLMs can be easily leveraged to generate deceptive misinformation at scale. Then, another important question is: how to combat LLM-generated misinformation? In this paper, we first systematically review the history of combating misinformation before the advent of LLMs. Then we illustrate the current efforts and present an outlook for these two fundamental questions respectively. The goal of this survey paper is to facilitate the progress of utilizing LLMs for fighting misinformation and call for interdisciplinary efforts from different stakeholders for combating LLM-generated misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2311.05596v1">LLM Augmented Hierarchical Agents</a></div>
    <div class="paper-meta">
      📅 2023-11-09
    </div>
    <details class="paper-abstract">
      Solving long-horizon, temporally-extended tasks using Reinforcement Learning (RL) is challenging, compounded by the common practice of learning without prior knowledge (or tabula rasa learning). Humans can generate and execute plans with temporally-extended actions and quickly learn to perform new tasks because we almost never solve problems from scratch. We want autonomous agents to have this same ability. Recently, LLMs have been shown to encode a tremendous amount of knowledge about the world and to perform impressive in-context learning and reasoning. However, using LLMs to solve real world problems is hard because they are not grounded in the current task. In this paper we exploit the planning capabilities of LLMs while using RL to provide learning from the environment, resulting in a hierarchical agent that uses LLMs to solve long-horizon tasks. Instead of completely relying on LLMs, they guide a high-level policy, making learning significantly more sample efficient. This approach is evaluated in simulation environments such as MiniGrid, SkillHack, and Crafter, and on a real robot arm in block manipulation tasks. We show that agents trained using our approach outperform other baselines methods and, once trained, don't need access to LLMs during deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.03744v2">Comparing Traditional and LLM-based Search for Consumer Choice: A Randomized Experiment</a></div>
    <div class="paper-meta">
      📅 2023-11-08
    </div>
    <details class="paper-abstract">
      Recent advances in the development of large language models are rapidly changing how online applications function. LLM-based search tools, for instance, offer a natural language interface that can accommodate complex queries and provide detailed, direct responses. At the same time, there have been concerns about the veracity of the information provided by LLM-based tools due to potential mistakes or fabrications that can arise in algorithmically generated text. In a set of online experiments we investigate how LLM-based search changes people's behavior relative to traditional search, and what can be done to mitigate overreliance on LLM-based output. Participants in our experiments were asked to solve a series of decision tasks that involved researching and comparing different products, and were randomly assigned to do so with either an LLM-based search tool or a traditional search engine. In our first experiment, we find that participants using the LLM-based tool were able to complete their tasks more quickly, using fewer but more complex queries than those who used traditional search. Moreover, these participants reported a more satisfying experience with the LLM-based search tool. When the information presented by the LLM was reliable, participants using the tool made decisions with a comparable level of accuracy to those using traditional search, however we observed overreliance on incorrect information when the LLM erred. Our second experiment further investigated this issue by randomly assigning some users to see a simple color-coded highlighting scheme to alert them to potentially incorrect or misleading information in the LLM responses. Overall we find that this confidence-based highlighting substantially increases the rate at which users spot incorrect information, improving the accuracy of their overall decisions while leaving most other measures unaffected.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05036v3">AvalonBench: Evaluating LLMs Playing the Game of Avalon</a></div>
    <div class="paper-meta">
      📅 2023-11-08
    </div>
    <details class="paper-abstract">
      In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged not only to make informed decisions based on dynamically evolving game phases, but also to engage in discussions where they must deceive, deduce, and negotiate with other players. These characteristics make Avalon a compelling test-bed to study the decision-making and language-processing capabilities of LLM Agents. To facilitate research in this line, we introduce AvalonBench - a comprehensive game environment tailored for evaluating multi-agent LLM Agents. This benchmark incorporates: (1) a game environment for Avalon, (2) rule-based bots as baseline opponents, and (3) ReAct-style LLM agents with tailored prompts for each role. Notably, our evaluations based on AvalonBench highlight a clear capability gap. For instance, models like ChatGPT playing good-role got a win rate of 22.2% against rule-based bots playing evil, while good-role bot achieves 38.2% win rate in the same setting. We envision AvalonBench could be a good test-bed for developing more advanced LLMs (with self-playing) and agent frameworks that can effectively model the layered complexities of such game environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18130v2">DELPHI: Data for Evaluating LLMs' Performance in Handling Controversial Issues</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 Accepted to EMNLP Industry Track 2023
    </div>
    <details class="paper-abstract">
      Controversy is a reflection of our zeitgeist, and an important aspect to any discourse. The rise of large language models (LLMs) as conversational systems has increased public reliance on these systems for answers to their various questions. Consequently, it is crucial to systematically examine how these models respond to questions that pertaining to ongoing debates. However, few such datasets exist in providing human-annotated labels reflecting the contemporary discussions. To foster research in this area, we propose a novel construction of a controversial questions dataset, expanding upon the publicly released Quora Question Pairs Dataset. This dataset presents challenges concerning knowledge recency, safety, fairness, and bias. We evaluate different LLMs using a subset of this dataset, illuminating how they handle controversial issues and the stances they adopt. This research ultimately contributes to our understanding of LLMs' interaction with controversial issues, paving the way for improvements in their comprehension and handling of complex societal debates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04177v1">Enhancing LLM Intelligence with ARM-RAG: Auxiliary Rationale Memory for Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are smart but forgetful. Recent studies, (e.g., (Bubeck et al., 2023)) on modern LLMs have shown that they are capable of performing amazing tasks typically necessitating human-level intelligence. However, unlike humans, frozen LLMs do not improve over time; they neither acquire new knowledge nor learn from their successes or failures. Some approaches to improving the intelligence of LLMs include fine-tuning models based on problem-solving performance (Zelikman et al., 2022), and building bigger and more sophisticated models (Bubeck et al., 2023). However, these methods have the drawback of requiring substantial data and computational resources to retrain existing models. In this paper, we explore the use of Retrieval Augmented Generation, also known as RAG (Lewis et al., 2021) to improve problem-solving performance. We propose ARM-RAG (Auxiliary Rationale Memory for Retrieval Augmented Generation), a system that learns from its successes without incurring high training costs. We demonstrate that the storage and subsequent retrieval of reasoning chains have a positive influence on performance in grade-school math problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04139v1">Modelling Sentiment Analysis: LLMs and data augmentation techniques</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 4 pages. For more information check the github link in the conclusion. Enjoy!
    </div>
    <details class="paper-abstract">
      This paper provides different approaches for a binary sentiment classification on a small training dataset. LLMs that provided state-of-the-art results in sentiment analysis and similar domains are being used, such as BERT, RoBERTa and XLNet.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03778v1">Bridging the Information Gap Between Domain-Specific Model and General LLM for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2023-11-07
    </div>
    <details class="paper-abstract">
      Generative large language models(LLMs) are proficient in solving general problems but often struggle to handle domain-specific tasks. This is because most of domain-specific tasks, such as personalized recommendation, rely on task-related information for optimal performance. Current methods attempt to supplement task-related information to LLMs by designing appropriate prompts or employing supervised fine-tuning techniques. Nevertheless, these methods encounter the certain issue that information such as community behavior pattern in RS domain is challenging to express in natural language, which limits the capability of LLMs to surpass state-of-the-art domain-specific models. On the other hand, domain-specific models for personalized recommendation which mainly rely on user interactions are susceptible to data sparsity due to their limited common knowledge capabilities. To address these issues, we proposes a method to bridge the information gap between the domain-specific models and the general large language models. Specifically, we propose an information sharing module which serves as an information storage mechanism and also acts as a bridge for collaborative training between the LLMs and domain-specific models. By doing so, we can improve the performance of LLM-based recommendation with the help of user behavior pattern information mined by domain-specific models. On the other hand, the recommendation performance of domain-specific models can also be improved with the help of common knowledge learned by LLMs. Experimental results on three real-world datasets have demonstrated the effectiveness of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03754v1">Which is better? Exploring Prompting Strategy For LLM-based Metrics</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 Eval4NLP 2023 shared task winner on both Small and Large model Track for Summarization
    </div>
    <details class="paper-abstract">
      This paper describes the DSBA submissions to the Prompting Large Language Models as Explainable Metrics shared task, where systems were submitted to two tracks: small and large summarization tracks. With advanced Large Language Models (LLMs) such as GPT-4, evaluating the quality of Natural Language Generation (NLG) has become increasingly paramount. Traditional similarity-based metrics such as BLEU and ROUGE have shown to misalign with human evaluation and are ill-suited for open-ended generation tasks. To address this issue, we explore the potential capability of LLM-based metrics, especially leveraging open-source LLMs. In this study, wide range of prompts and prompting techniques are systematically analyzed with three approaches: prompting strategy, score aggregation, and explainability. Our research focuses on formulating effective prompt templates, determining the granularity of NLG quality scores and assessing the impact of in-context examples on LLM-based evaluation. Furthermore, three aggregation strategies are compared to identify the most reliable method for aggregating NLG quality scores. To examine explainability, we devise a strategy that generates rationales for the scores and analyzes the characteristics of the explanation produced by the open-source LLMs. Extensive experiments provide insights regarding evaluation capabilities of open-source LLMs and suggest effective prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03716v1">LLM as an Art Director (LaDi): Using LLMs to improve Text-to-Media Generators</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 12 pages, System Demonstration/Industry paper. Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in text-to-image generation have revolutionized numerous fields, including art and cinema, by automating the generation of high-quality, context-aware images and video. However, the utility of these technologies is often limited by the inadequacy of text prompts in guiding the generator to produce artistically coherent and subject-relevant images. In this paper, We describe the techniques that can be used to make Large Language Models (LLMs) act as Art Directors that enhance image and video generation. We describe our unified system for this called "LaDi". We explore how LaDi integrates multiple techniques for augmenting the capabilities of text-to-image generators (T2Is) and text-to-video generators (T2Vs), with a focus on constrained decoding, intelligent prompting, fine-tuning, and retrieval. LaDi and these techniques are being used today in apps and platforms developed by Plai Labs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.04657v3">BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset</a></div>
    <div class="paper-meta">
      📅 2023-11-07
      | 💬 Published at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      In this paper, we introduce the BeaverTails dataset, aimed at fostering research on safety alignment in large language models (LLMs). This dataset uniquely separates annotations of helpfulness and harmlessness for question-answering pairs, thus offering distinct perspectives on these crucial attributes. In total, we have gathered safety meta-labels for 333,963 question-answer (QA) pairs and 361,903 pairs of expert comparison data for both the helpfulness and harmlessness metrics. We further showcase applications of BeaverTails in content moderation and reinforcement learning with human feedback (RLHF), emphasizing its potential for practical safety measures in LLMs. We believe this dataset provides vital resources for the community, contributing towards the safe development and deployment of LLMs. Our project page is available at the following URL: https://sites.google.com/view/pku-beavertails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.03243v1">Safurai-Csharp: Harnessing Synthetic Data to improve language-specific Code LLM</a></div>
    <div class="paper-meta">
      📅 2023-11-06
    </div>
    <details class="paper-abstract">
      This paper introduces Safurai-Csharp, an open-source model designed to specialize in the generation, completion, and debugging of C# code. Safurai-Csharp is built upon the novel CodeLlama 34B model and leverages the EvolInstruct technique, creating a refined and expanded dataset for its fine-tuning process. The results of its performance, a notable score of 56.33% on the Manual MultiPL-E benchmark (Zero-Shot, Pass@1), signal its high capacity to streamline developers' workflows and aid code learning. It shows promise in setting new stakes in the landscape of open-source C# LLMs and hopes to inspire more inclusive and wide-ranging development in the field of language-specific LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2303.03480v2">Can an Embodied Agent Find Your "Cat-shaped Mug"? LLM-Guided Exploration for Zero-Shot Object Navigation</a></div>
    <div class="paper-meta">
      📅 2023-11-05
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      We present LGX (Language-guided Exploration), a novel algorithm for Language-Driven Zero-Shot Object Goal Navigation (L-ZSON), where an embodied agent navigates to a uniquely described target object in a previously unseen environment. Our approach makes use of Large Language Models (LLMs) for this task by leveraging the LLM's commonsense reasoning capabilities for making sequential navigational decisions. Simultaneously, we perform generalized target object detection using a pre-trained Vision-Language grounding model. We achieve state-of-the-art zero-shot object navigation results on RoboTHOR with a success rate (SR) improvement of over 27% over the current baseline of the OWL-ViT CLIP on Wheels (OWL CoW). Furthermore, we study the usage of LLMs for robot navigation and present an analysis of various prompting strategies affecting the model output. Finally, we showcase the benefits of our approach via \textit{real-world} experiments that indicate the superior performance of LGX in detecting and navigating to visually unique objects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.14202v2">Fine-tuned LLMs Know More, Hallucinate Less with Few-Shot Sequence-to-Sequence Semantic Parsing over Wikidata</a></div>
    <div class="paper-meta">
      📅 2023-11-05
      | 💬 EMNLP 2023 Main
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) can answer many questions correctly, they can also hallucinate and give wrong answers. Wikidata, with its over 12 billion facts, can be used to ground LLMs to improve their factuality. This paper presents WikiWebQuestions, a high-quality question answering benchmark for Wikidata. Ported over from WebQuestions for Freebase, it consists of real-world data with SPARQL annotation. This paper presents a few-shot sequence-to-sequence semantic parser for Wikidata. We modify SPARQL to use the unique domain and property names instead of their IDs. We train the parser to use either the results from an entity linker or mentions in the query. We fine-tune LLaMA by adding the few-shot training data to that used to fine-tune Alpaca. Our experimental results demonstrate the effectiveness of this methodology, establishing a strong baseline of 76% and 65% answer accuracy in the dev and test sets of WikiWebQuestions, respectively. By pairing our semantic parser with GPT-3, we combine verifiable results with qualified GPT-3 guesses to provide useful answers to 96% of the questions in dev. We also show that our method outperforms the state-of-the-art for the QALD-7 Wikidata dataset by 3.6% in F1 score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02660v1">LLM-enhanced Self-training for Cross-domain Constituency Parsing</a></div>
    <div class="paper-meta">
      📅 2023-11-05
      | 💬 Accepted by EMNLP 2023 main conf
    </div>
    <details class="paper-abstract">
      Self-training has proven to be an effective approach for cross-domain tasks, and in this study, we explore its application to cross-domain constituency parsing. Traditional self-training methods rely on limited and potentially low-quality raw corpora. To overcome this limitation, we propose enhancing self-training with the large language model (LLM) to generate domain-specific raw corpora iteratively. For the constituency parsing, we introduce grammar rules that guide the LLM in generating raw corpora and establish criteria for selecting pseudo instances. Our experimental results demonstrate that self-training for constituency parsing, equipped with an LLM, outperforms traditional methods regardless of the LLM's performance. Moreover, the combination of grammar rules and confidence criteria for pseudo-data selection yields the highest performance in the cross-domain constituency parsing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02597v1">FloodBrain: Flood Disaster Reporting by Web-based Retrieval Augmented Generation with an LLM</a></div>
    <div class="paper-meta">
      📅 2023-11-05
      | 💬 Version is the one submitted to Artificial Intelligence for Humanitarian Assistance and Disaster Response Workshop @Neurips2023. All authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Fast disaster impact reporting is crucial in planning humanitarian assistance. Large Language Models (LLMs) are well known for their ability to write coherent text and fulfill a variety of tasks relevant to impact reporting, such as question answering or text summarization. However, LLMs are constrained by the knowledge within their training data and are prone to generating inaccurate, or "hallucinated", information. To address this, we introduce a sophisticated pipeline embodied in our tool FloodBrain (floodbrain.com), specialized in generating flood disaster impact reports by extracting and curating information from the web. Our pipeline assimilates information from web search results to produce detailed and accurate reports on flood events. We test different LLMs as backbones in our tool and compare their generated reports to human-written reports on different metrics. Similar to other studies, we find a notable correlation between the scores assigned by GPT-4 and the scores given by human evaluators when comparing our generated reports to human-authored ones. Additionally, we conduct an ablation study to test our single pipeline components and their relevancy for the final reports. With our tool, we aim to advance the use of LLMs for disaster impact reporting and reduce the time for coordination of humanitarian efforts in the wake of flood disasters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.09812v2">R2GenGPT: Radiology Report Generation with Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-05
      | 💬 Accepted by meta-radiology
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have consistently showcased remarkable generalization capabilities when applied to various language tasks. Nonetheless, harnessing the full potential of LLMs for Radiology Report Generation (R2Gen) still presents a challenge, stemming from the inherent disparity in modality between LLMs and the R2Gen task. To bridge this gap effectively, we propose R2GenGPT, which is a novel solution that aligns visual features with the word embedding space of LLMs using an efficient visual alignment module. This innovative approach empowers the previously static LLM to seamlessly integrate and process image information, marking a step forward in optimizing R2Gen performance. R2GenGPT offers the following benefits. First, it attains state-of-the-art (SOTA) performance by training only the lightweight visual alignment module while freezing all the parameters of LLM. Second, it exhibits high training efficiency, as it requires the training of an exceptionally minimal number of parameters while achieving rapid convergence. By employing delta tuning, our model only trains 5M parameters (which constitute just 0.07\% of the total parameter count) to achieve performance close to the SOTA levels. Our code is available at https://github.com/wang-zhanyu/R2GenGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02303v1">MFTCoder: Boosting Code LLMs with Multitask Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2023-11-04
    </div>
    <details class="paper-abstract">
      Code LLMs have emerged as a specialized research field, with remarkable studies dedicated to enhancing model's coding capabilities through fine-tuning on pre-trained models. Previous fine-tuning approaches were typically tailored to specific downstream tasks or scenarios, which meant separate fine-tuning for each task, requiring extensive training resources and posing challenges in terms of deployment and maintenance. Furthermore, these approaches failed to leverage the inherent interconnectedness among different code-related tasks. To overcome these limitations, we present a multi-task fine-tuning framework, MFTcoder, that enables simultaneous and parallel fine-tuning on multiple tasks. By incorporating various loss functions, we effectively address common challenges in multi-task learning, such as data imbalance, varying difficulty levels, and inconsistent convergence speeds. Extensive experiments have conclusively demonstrated that our multi-task fine-tuning approach outperforms both individual fine-tuning on single tasks and fine-tuning on a mixed ensemble of tasks. Moreover, MFTcoder offers efficient training capabilities, including efficient data tokenization modes and PEFT fine-tuning, resulting in significantly improved speed compared to traditional fine-tuning methods. MFTcoder seamlessly integrates with several mainstream open-source LLMs, such as CodeLLama and Qwen. Leveraging the CodeLLama foundation, our MFTcoder fine-tuned model, \textsc{CodeFuse-CodeLLama-34B}, achieves an impressive pass@1 score of 74.4\% on the HumaneEval benchmark, surpassing GPT-4 performance (67\%, zero-shot). MFTCoder is open-sourced at \url{https://github.com/codefuse-ai/MFTCOder}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02294v1">LLMs grasp morality in concept</a></div>
    <div class="paper-meta">
      📅 2023-11-04
      | 💬 Presented at NeurIPS 2023 Moral Pyschology and Moral Philosophy workshop
    </div>
    <details class="paper-abstract">
      Work in AI ethics and fairness has made much progress in regulating LLMs to reflect certain values, such as fairness, truth, and diversity. However, it has taken the problem of how LLMs might 'mean' anything at all for granted. Without addressing this, it is not clear what imbuing LLMs with such values even means. In response, we provide a general theory of meaning that extends beyond humans. We use this theory to explicate the precise nature of LLMs as meaning-agents. We suggest that the LLM, by virtue of its position as a meaning-agent, already grasps the constructions of human society (e.g. morality, gender, and race) in concept. Consequently, under certain ethical frameworks, currently popular methods for model alignment are limited at best and counterproductive at worst. Moreover, unaligned models may help us better develop our moral and social philosophy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02268v1">LLMs-augmented Contextual Bandit</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 Accepted by the Foundation Models for Decision Making workshop at NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Contextual bandits have emerged as a cornerstone in reinforcement learning, enabling systems to make decisions with partial feedback. However, as contexts grow in complexity, traditional bandit algorithms can face challenges in adequately capturing and utilizing such contexts. In this paper, we propose a novel integration of large language models (LLMs) with the contextual bandit framework. By leveraging LLMs as an encoder, we enrich the representation of the context, providing the bandit with a denser and more informative view. Preliminary results on synthetic datasets demonstrate the potential of this approach, showing notable improvements in cumulative rewards and reductions in regret compared to traditional bandit algorithms. This integration not only showcases the capabilities of LLMs in reinforcement learning but also opens the door to a new era of contextually-aware decision systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02049v1">Post Turing: Mapping the landscape of LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 Accepted for GEM @ EMNLP 2023
    </div>
    <details class="paper-abstract">
      In the rapidly evolving landscape of Large Language Models (LLMs), introduction of well-defined and standardized evaluation methodologies remains a crucial challenge. This paper traces the historical trajectory of LLM evaluations, from the foundational questions posed by Alan Turing to the modern era of AI research. We categorize the evolution of LLMs into distinct periods, each characterized by its unique benchmarks and evaluation criteria. As LLMs increasingly mimic human-like behaviors, traditional evaluation proxies, such as the Turing test, have become less reliable. We emphasize the pressing need for a unified evaluation system, given the broader societal implications of these models. Through an analysis of common evaluation methodologies, we advocate for a qualitative shift in assessment approaches, underscoring the importance of standardization and objective criteria. This work serves as a call for the AI community to collaboratively address the challenges of LLM evaluation, ensuring their reliability, fairness, and societal benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.02618v2">ChatGPT for GTFS: Benchmarking LLMs on GTFS Understanding and Retrieval</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 22 pages, 8 figures, 1 table, Public Transport
    </div>
    <details class="paper-abstract">
      The General Transit Feed Specification (GTFS) standard for publishing transit data is ubiquitous. GTFS being tabular data, with information spread across different files, necessitates specialized tools or packages to retrieve information. Concurrently, the use of Large Language Models(LLMs) for text and information retrieval is growing. The idea of this research is to see if the current widely adopted LLMs (ChatGPT) are able to understand GTFS and retrieve information from GTFS using natural language instructions without explicitly providing information. In this research, we benchmark OpenAI's GPT-3.5-Turbo and GPT-4 LLMs which are the backbone of ChatGPT. ChatGPT demonstrates a reasonable understanding of GTFS by answering 59.7% (GPT-3.5-Turbo) and 73.3% (GPT-4) of our multiple-choice questions (MCQ) correctly. Furthermore, we evaluated the LLMs on information extraction tasks using a filtered GTFS feed containing four routes. We found that program synthesis techniques outperformed zero-shot approaches, achieving up to 93% (90%) accuracy for simple queries and 61% (41%) for complex ones using GPT-4 (GPT-3.5-Turbo).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2304.04370v6">OpenAGI: When LLM Meets Domain Experts</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 In NeurIPS 2023
    </div>
    <details class="paper-abstract">
      Human Intelligence (HI) excels at combining basic skills to solve complex tasks. This capability is vital for Artificial Intelligence (AI) and should be embedded in comprehensive AI Agents, enabling them to harness expert models for complex task-solving towards Artificial General Intelligence (AGI). Large Language Models (LLMs) show promising learning and reasoning abilities, and can effectively use external models, tools, plugins, or APIs to tackle complex problems. In this work, we introduce OpenAGI, an open-source AGI research and development platform designed for solving multi-step, real-world tasks. Specifically, OpenAGI uses a dual strategy, integrating standard benchmark tasks for benchmarking and evaluation, and open-ended tasks including more expandable models, tools, plugins, or APIs for creative problem-solving. Tasks are presented as natural language queries to the LLM, which then selects and executes appropriate models. We also propose a Reinforcement Learning from Task Feedback (RLTF) mechanism that uses task results to improve the LLM's task-solving ability, which creates a self-improving AI feedback loop. While we acknowledge that AGI is a broad and multifaceted research challenge with no singularly defined solution path, the integration of LLMs with domain-specific expert models, inspired by mirroring the blend of general and specialized intelligence in humans, offers a promising approach towards AGI. We are open-sourcing the OpenAGI project's code, dataset, benchmarks, evaluation methods, and the UI demo to foster community involvement in AGI advancement: https://github.com/agiresearch/OpenAGI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01964v1">Don't Make Your LLM an Evaluation Benchmark Cheater</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Large language models~(LLMs) have greatly advanced the frontiers of artificial intelligence, attaining remarkable improvement in model capacity. To assess the model performance, a typical approach is to construct evaluation benchmarks for measuring the ability level of LLMs in different aspects. Despite that a number of high-quality benchmarks have been released, the concerns about the appropriate use of these benchmarks and the fair comparison of different models are increasingly growing. Considering these concerns, in this paper, we discuss the potential risk and impact of inappropriately using evaluation benchmarks and misleadingly interpreting the evaluation results. Specially, we focus on a special issue that would lead to inappropriate evaluation, \ie \emph{benchmark leakage}, referring that the data related to evaluation sets is occasionally used for model training. This phenomenon now becomes more common since pre-training data is often prepared ahead of model test. We conduct extensive experiments to study the effect of benchmark leverage, and find that it can dramatically boost the evaluation results, which would finally lead to an unreliable assessment of model performance. To improve the use of existing evaluation benchmarks, we finally present several guidelines for both LLM developers and benchmark maintainers. We hope this work can draw attention to appropriate training and evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01876v1">Sentiment Analysis through LLM Negotiations</a></div>
    <div class="paper-meta">
      📅 2023-11-03
      | 💬 Pre-print Version
    </div>
    <details class="paper-abstract">
      A standard paradigm for sentiment analysis is to rely on a singular LLM and makes the decision in a single round under the framework of in-context learning. This framework suffers the key disadvantage that the single-turn output generated by a single LLM might not deliver the perfect decision, just as humans sometimes need multiple attempts to get things right. This is especially true for the task of sentiment analysis where deep reasoning is required to address the complex linguistic phenomenon (e.g., clause composition, irony, etc) in the input. To address this issue, this paper introduces a multi-LLM negotiation framework for sentiment analysis. The framework consists of a reasoning-infused generator to provide decision along with rationale, a explanation-deriving discriminator to evaluate the credibility of the generator. The generator and the discriminator iterate until a consensus is reached. The proposed framework naturally addressed the aforementioned challenge, as we are able to take the complementary abilities of two LLMs, have them use rationale to persuade each other for correction. Experiments on a wide range of sentiment analysis benchmarks (SST-2, Movie Review, Twitter, yelp, amazon, IMDB) demonstrate the effectiveness of proposed approach: it consistently yields better performances than the ICL baseline across all benchmarks, and even superior performances to supervised baselines on the Twitter and movie review datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.02126v1">PILL: Plug Into LLM with Adapter Expert and Attention Gate</a></div>
    <div class="paper-meta">
      📅 2023-11-03
    </div>
    <details class="paper-abstract">
      Due to the remarkable capabilities of powerful Large Language Models (LLMs) in effectively following instructions, there has been a growing number of assistants in the community to assist humans. Recently, significant progress has been made in the development of Vision Language Models (VLMs), expanding the capabilities of LLMs and enabling them to execute more diverse instructions. However, it is foreseeable that models will likely need to handle tasks involving additional modalities such as speech, video, and others. This poses a particularly prominent challenge of dealing with the complexity of mixed modalities. To address this, we introduce a novel architecture called PILL: Plug Into LLM with adapter expert and attention gate to better decouple these complex modalities and leverage efficient fine-tuning. We introduce two modules: Firstly, utilizing Mixture-of-Modality-Adapter-Expert to independently handle different modalities, enabling better adaptation to downstream tasks while preserving the expressive capability of the original model. Secondly, by introducing Modality-Attention-Gating, which enables adaptive control of the contribution of modality tokens to the overall representation. In addition, we have made improvements to the Adapter to enhance its learning and expressive capabilities. Experimental results demonstrate that our approach exhibits competitive performance compared to other mainstream methods for modality fusion. For researchers interested in our work, we provide free access to the code and models at https://github.com/DsaltYfish/PILL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01792v1">AFPQ: Asymmetric Floating Point Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great performance in various tasks, but face deployment challenges from limited memory capacity and bandwidth. Low-bit weight quantization can save memory and accelerate inference. Although floating-point (FP) formats show good performance in LLM quantization, they tend to perform poorly with small group sizes or sub-4 bits. We find the reason is that the absence of asymmetry in previous FP quantization makes it unsuitable for handling asymmetric value distribution of LLM weight tensors. In this work, we propose asymmetric FP quantization (AFPQ), which sets separate scales for positive and negative values. Our method leads to large accuracy improvements and can be easily plugged into other quantization methods, including GPTQ and AWQ, for better performance. Besides, no additional storage is needed compared with asymmetric integer (INT) quantization. The code is available at https://github.com/zhangsichengsjtu/AFPQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.01456v2">LLM and Infrastructure as a Code use case</a></div>
    <div class="paper-meta">
      📅 2023-11-02
    </div>
    <details class="paper-abstract">
      Cloud computing and the evolution of management methodologies such as Lean Management or Agile entail a profound transformation in both system construction and maintenance approaches. These practices are encompassed within the term "DevOps." This descriptive approach to an information system or application, alongside the configuration of its constituent components, has necessitated the development of descriptive languages paired with specialized engines for automating systems administration tasks. Among these, the tandem of Ansible (engine) and YAML (descriptive language) stands out as the two most prevalent tools in the market, facing notable competition mainly from Terraform. The current document presents an inquiry into a solution for generating and managing Ansible YAML roles and playbooks, utilizing Generative LLMs (Language Models) to translate human descriptions into code. Our efforts are focused on identifying plausible directions and outlining the potential industrial applications. Note: For the purpose of this experiment, we have opted against the use of Ansible Lightspeed. This is due to its reliance on an IBM Watson model, for which we have not found any publicly available references. Comprehensive information regarding this remarkable technology can be found [1] directly on our partner's website, RedHat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.01049v1">Multi-dimensional data refining strategy for effective fine-tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-02
    </div>
    <details class="paper-abstract">
      Data is a cornerstone for fine-tuning large language models, yet acquiring suitable data remains challenging. Challenges encompassed data scarcity, linguistic diversity, and domain-specific content. This paper presents lessons learned while crawling and refining data tailored for fine-tuning Vietnamese language models. Crafting such a dataset, while accounting for linguistic intricacies and striking a balance between inclusivity and accuracy, demands meticulous planning. Our paper presents a multidimensional strategy including leveraging existing datasets in the English language and developing customized data-crawling scripts with the assistance of generative AI tools. A fine-tuned LLM model for the Vietnamese language, which was produced using resultant datasets, demonstrated good performance while generating Vietnamese news articles from prompts. The study offers practical solutions and guidance for future fine-tuning models in languages like Vietnamese.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13229v2">The GitHub Recent Bugs Dataset for Evaluating LLM-based Debugging Applications</a></div>
    <div class="paper-meta">
      📅 2023-11-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong natural language processing and code synthesis capabilities, which has led to their rapid adoption in software engineering applications. However, details about LLM training data are often not made public, which has caused concern as to whether existing bug benchmarks are included. In lieu of the training data for the popular GPT models, we examine the training data of the open-source LLM StarCoder, and find it likely that data from the widely used Defects4J benchmark was included, raising the possibility of its inclusion in GPT training data as well. This makes it difficult to tell how well LLM-based results on Defects4J would generalize, as for any results it would be unclear whether a technique's performance is due to LLM generalization or memorization. To remedy this issue and facilitate continued research on LLM-based SE, we present the GitHub Recent Bugs (GHRB) dataset, which includes 76 real-world Java bugs that were gathered after the OpenAI data cut-off point.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00681v1">Are Large Language Models Reliable Judges? A Study on the Factuality Evaluation Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-01
      | 💬 accepted by Generation, Evaluation & Metrics (GEM) Workshop at EMNLP 2023
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have gained immense attention due to their notable emergent capabilities, surpassing those seen in earlier language models. A particularly intriguing application of LLMs is their role as evaluators for texts produced by various generative models. In this study, we delve into the potential of LLMs as reliable assessors of factual consistency in summaries generated by text-generation models. Initially, we introduce an innovative approach for factuality assessment using LLMs. This entails employing a singular LLM for the entirety of the question-answering-based factuality scoring process. Following this, we examine the efficacy of various LLMs in direct factuality scoring, benchmarking them against traditional measures and human annotations. Contrary to initial expectations, our results indicate a lack of significant correlations between factuality metrics and human evaluations, specifically for GPT-4 and PaLM-2. Notable correlations were only observed with GPT-3.5 across two factuality subcategories. These consistent findings across various factual error categories suggest a fundamental limitation in the current LLMs' capability to accurately gauge factuality. This version presents the information more concisely while maintaining the main points and findings of the original text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.13976v2">Advancing Requirements Engineering through Generative AI: Assessing the Role of LLMs</a></div>
    <div class="paper-meta">
      📅 2023-11-01
    </div>
    <details class="paper-abstract">
      Requirements Engineering (RE) is a critical phase in software development including the elicitation, analysis, specification, and validation of software requirements. Despite the importance of RE, it remains a challenging process due to the complexities of communication, uncertainty in the early stages and inadequate automation support. In recent years, large-language models (LLMs) have shown significant promise in diverse domains, including natural language processing, code generation, and program understanding. This chapter explores the potential of LLMs in driving RE processes, aiming to improve the efficiency and accuracy of requirements-related tasks. We propose key directions and SWOT analysis for research and development in using LLMs for RE, focusing on the potential for requirements elicitation, analysis, specification, and validation. We further present the results from a preliminary evaluation, in this context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00306v1">Probing Explicit and Implicit Gender Bias through LLM Conditional Text Generation</a></div>
    <div class="paper-meta">
      📅 2023-11-01
      | 💬 Accepted in Socially Responsible Language Modelling Research (SoLaR) 2023 at NeurIPS 2023; the first two authors contribute equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate biased and toxic responses. Yet most prior work on LLM gender bias evaluation requires predefined gender-related phrases or gender stereotypes, which are challenging to be comprehensively collected and are limited to explicit bias evaluation. In addition, we believe that instances devoid of gender-related language or explicit stereotypes in inputs can still induce gender bias in LLMs. Thus, in this work, we propose a conditional text generation mechanism without the need for predefined gender phrases and stereotypes. This approach employs three types of inputs generated through three distinct strategies to probe LLMs, aiming to show evidence of explicit and implicit gender biases in LLMs. We also utilize explicit and implicit evaluation metrics to evaluate gender bias in LLMs under different strategies. Our experiments demonstrate that an increased model size does not consistently lead to enhanced fairness and all tested LLMs exhibit explicit and/or implicit gender bias, even when explicit gender stereotypes are absent in the inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00273v1">SoulChat: Improving LLMs' Empathy, Listening, and Comfort Abilities through Fine-tuning with Multi-turn Empathy Conversations</a></div>
    <div class="paper-meta">
      📅 2023-11-01
      | 💬 Appectped to Findings of EMNLP2023
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely applied in various fields due to their excellent capability for memorizing knowledge and chain of thought (CoT). When these language models are applied in the field of psychological counseling, they often rush to provide universal advice. However, when users seek psychological support, they need to gain empathy, trust, understanding and comfort, rather than just reasonable advice. To this end, we constructed a multi-turn empathetic conversation dataset of more than 2 million samples, in which the input is the multi-turn conversation context, and the target is empathetic responses that cover expressions such as questioning, comfort, recognition, listening, trust, emotional support, etc. Experiments have shown that the empathy ability of LLMs can be significantly enhanced when finetuning by using multi-turn dialogue history and responses that are closer to the expression of a psychological consultant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.00272v1">ChatCoder: Chat-based Refine Requirement Improves LLMs' Code Generation</a></div>
    <div class="paper-meta">
      📅 2023-11-01
    </div>
    <details class="paper-abstract">
      Large language models have shown good performances in generating code to meet human requirements. However, human requirements expressed in natural languages can be vague, incomplete, and ambiguous, leading large language models to misunderstand human requirements and make mistakes. Worse, it is difficult for a human user to refine the requirement. To help human users refine their requirements and improve large language models' code generation performances, we propose ChatCoder: a method to refine the requirements via chatting with large language models. We design a chat scheme in which the large language models will guide the human users to refine their expression of requirements to be more precise, unambiguous, and complete than before. Experiments show that ChatCoder has improved existing large language models' performance by a large margin. Besides, ChatCoder has the advantage over refine-based methods and LLMs fine-tuned via human response.
    </details>
</div>
