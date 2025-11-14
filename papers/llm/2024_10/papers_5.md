# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13640v1">Latent Space Chain-of-Embedding Enables Output-free LLM Self-Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 33 pages, 18 figures, 12 tables
    </div>
    <details class="paper-abstract">
      LLM self-evaluation relies on the LLM's own ability to estimate response correctness, which can greatly improve its deployment reliability. In this research track, we propose the Chain-of-Embedding (CoE) in the latent space to enable LLMs to perform output-free self-evaluation. CoE consists of all progressive hidden states produced during the inference time, which can be treated as the latent thinking path of LLMs. We find that when LLMs respond correctly and incorrectly, their CoE features differ, these discrepancies assist us in estimating LLM response correctness. Experiments in four diverse domains and seven LLMs fully demonstrate the effectiveness of our method. Meanwhile, its label-free design intent without any training and millisecond-level computational cost ensure real-time feedback in large-scale scenarios. More importantly, we provide interesting insights into LLM response correctness from the perspective of hidden state changes inside LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10398v2">FairMindSim: Alignment of Behavior, Emotion, and Belief in Humans and LLM Agents Amid Ethical Dilemmas</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      AI alignment is a pivotal issue concerning AI control and safety. It should consider not only value-neutral human preferences but also moral and ethical considerations. In this study, we introduced FairMindSim, which simulates the moral dilemma through a series of unfair scenarios. We used LLM agents to simulate human behavior, ensuring alignment across various stages. To explore the various socioeconomic motivations, which we refer to as beliefs, that drive both humans and LLM agents as bystanders to intervene in unjust situations involving others, and how these beliefs interact to influence individual behavior, we incorporated knowledge from relevant sociological fields and proposed the Belief-Reward Alignment Behavior Evolution Model (BREM) based on the recursive reward model (RRM). Our findings indicate that, behaviorally, GPT-4o exhibits a stronger sense of social justice, while humans display a richer range of emotions. Additionally, we discussed the potential impact of emotions on behavior. This study provides a theoretical foundation for applications in aligning LLMs with altruistic values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13610v1">MeNTi: Bridging Medical Calculator and LLM Agent with Nested Tool Calling</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Integrating tools into Large Language Models (LLMs) has facilitated the widespread application. Despite this, in specialized downstream task contexts, reliance solely on tools is insufficient to fully address the complexities of the real world. This particularly restricts the effective deployment of LLMs in fields such as medicine. In this paper, we focus on the downstream tasks of medical calculators, which use standardized tests to assess an individual's health status. We introduce MeNTi, a universal agent architecture for LLMs. MeNTi integrates a specialized medical toolkit and employs meta-tool and nested calling mechanisms to enhance LLM tool utilization. Specifically, it achieves flexible tool selection and nested tool calling to address practical issues faced in intricate medical scenarios, including calculator selection, slot filling, and unit conversion. To assess the capabilities of LLMs for quantitative assessment throughout the clinical process of calculator scenarios, we introduce CalcQA. This benchmark requires LLMs to use medical calculators to perform calculations and assess patient health status. CalcQA is constructed by professional physicians and includes 100 case-calculator pairs, complemented by a toolkit of 281 medical tools. The experimental results demonstrate significant performance improvements with our framework. This research paves new directions for applying LLMs in demanding scenarios of medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13542v1">LLM-based Unit Test Generation via Property Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Automated unit test generation has been widely studied, with Large Language Models (LLMs) recently showing significant potential. Moreover, in the context of unit test generation, these tools prioritize high code coverage, often at the expense of practical usability, correctness, and maintainability. In response, we propose Property-Based Retrieval Augmentation, a novel mechanism that extends LLM-based Retrieval-Augmented Generation (RAG) beyond basic vector, text similarity, and graph-based methods. Our approach considers task-specific context and introduces a tailored property retrieval mechanism. Specifically, in the unit test generation task, we account for the unique structure of unit tests by dividing the test generation process into Given, When, and Then phases. When generating tests for a focal method, we not only retrieve general context for the code under test but also consider task-specific context such as pre-existing tests of other methods, which can provide valuable insights for any of the Given, When, and Then phases. This forms property relationships between focal method and other methods, thereby expanding the scope of retrieval beyond traditional RAG. We implement this approach in a tool called APT, which sequentially performs preprocessing, property retrieval, and unit test generation, using an iterative strategy where newly generated tests guide the creation of subsequent ones. We evaluated APT on 12 open-source projects with 1515 methods, and the results demonstrate that APT consistently outperforms existing tools in terms of correctness, completeness, and maintainability of the generated tests. Moreover, we introduce a novel code-context-aware retrieval mechanism for LLMs beyond general context, offering valuable insights and potential applications for other code-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12294v2">LLM-based Cognitive Models of Students with Misconceptions</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Accurately modeling student cognition is crucial for developing effective AI-driven educational technologies. A key challenge is creating realistic student models that satisfy two essential properties: (1) accurately replicating specific misconceptions, and (2) correctly solving problems where these misconceptions are not applicable. This dual requirement reflects the complex nature of student understanding, where misconceptions coexist with correct knowledge. This paper investigates whether Large Language Models (LLMs) can be instruction-tuned to meet this dual requirement and effectively simulate student thinking in algebra. We introduce MalAlgoPy, a novel Python library that generates datasets reflecting authentic student solution patterns through a graph-based representation of algebraic problem-solving. Utilizing MalAlgoPy, we define and examine Cognitive Student Models (CSMs) - LLMs instruction tuned to faithfully emulate realistic student behavior. Our findings reveal that LLMs trained on misconception examples can efficiently learn to replicate errors. However, the training diminishes the model's ability to solve problems correctly, particularly for problem types where the misconceptions are not applicable, thus failing to satisfy second property of CSMs. We demonstrate that by carefully calibrating the ratio of correct to misconception examples in the training data - sometimes as low as 0.25 - it is possible to develop CSMs that satisfy both properties. Our insights enhance our understanding of AI-based student models and pave the way for effective adaptive learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17453v3">Learning to Ask Informative Questions: Enhancing LLMs with Preference Optimization and Expected Information Gain</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted to EMNLP 2024 (Findings)
    </div>
    <details class="paper-abstract">
      Questions are essential tools for acquiring the necessary information to complete information-seeking tasks. However, large language models (LLMs), especially open-source models, often perform poorly in generating informative questions, as measured by expected information gain (EIG). In this paper, we propose a method to enhance the informativeness of LLM-generated questions in 20-question game dialogues. We sample multiple questions from the same model (LLAMA 2-CHAT 7B) for each game and create pairs of low-EIG and high-EIG questions to apply a Direct Preference Optimization (DPO) algorithm. Our results show that this method produces more effective questions (in terms of EIG), even in domains different from those used to train the DPO model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13461v1">Progressive Mixed-Precision Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      In spite of the great potential of large language models (LLMs) across various tasks, their deployment on resource-constrained devices remains challenging due to their excessive computational and memory demands. Quantization has emerged as an effective solution by storing weights in reduced precision. However, utilizing low precisions (i.e.~2/3-bit) to substantially alleviate the memory-boundedness of LLM decoding, still suffers from prohibitive performance drop. In this work, we argue that existing approaches fail to explore the diversity in computational patterns, redundancy, and sensitivity to approximations of the different phases of LLM inference, resorting to a uniform quantization policy throughout. Instead, we propose a novel phase-aware method that selectively allocates precision during different phases of LLM inference, achieving both strong context extraction during prefill and efficient memory bandwidth utilization during decoding. To further address the memory-boundedness of the decoding phase, we introduce Progressive Mixed-Precision Decoding (PMPD), a technique that enables the gradual lowering of precision deeper in the generated sequence, together with a spectrum of precision-switching schedulers that dynamically drive the precision-lowering decisions in either task-adaptive or prompt-adaptive manner. Extensive evaluation across diverse language tasks shows that when targeting Nvidia GPUs, PMPD achieves 1.4$-$12.2$\times$ speedup in matrix-vector multiplications over fp16 models, while when targeting an LLM-optimized NPU, our approach delivers a throughput gain of 3.8$-$8.0$\times$ over fp16 models and up to 1.54$\times$ over uniform quantization approaches while preserving the output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12475v2">Aegis:An Advanced LLM-Based Multi-Agent for Intelligent Functional Safety Engineering</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Functional safety is a critical aspect of automotive engineering, encompassing all phases of a vehicle's lifecycle, including design, development, production, operation, and decommissioning. This domain involves highly knowledge-intensive tasks. This paper introduces Aegis: An Advanced LLM-Based Multi-Agent for Intelligent Functional Safety Engineering. Aegis is specifically designed to support complex functional safety tasks within the automotive sector. It is tailored to perform Hazard Analysis and Risk Assessment(HARA), document Functional Safety Requirements(FSR), and plan test cases for Automatic Emergency Braking(AEB) systems. The most advanced version, Aegis-Max, leverages Retrieval-Augmented Generation(RAG) and reflective mechanisms to enhance its capability in managing complex, knowledge-intensive tasks. Additionally, targeted prompt refinement by professional functional safety practitioners can significantly optimize Aegis's performance in the functional safety domain. This paper demonstrates the potential of Aegis to improve the efficiency and effectiveness of functional safety processes in automotive engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13394v1">Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Evaluating machine-generated text remains a significant challenge in NLP, especially for non-English languages. Current methodologies, including automated metrics, human assessments, and LLM-based evaluations, predominantly focus on English, revealing a significant gap in multilingual evaluation frameworks. We introduce the Cross Lingual Auto Evaluation (CIA) Suite, an extensible framework that includes evaluator LLMs (Hercule) and a novel test set (Recon) specifically designed for multilingual evaluation. Our test set features 500 human-annotated instructions spanning various task capabilities along with human judgment scores across six languages. This would enable benchmarking of general-purpose multilingual LLMs and facilitate meta-evaluation of Evaluator LLMs. The proposed model, Hercule, is a cross-lingual evaluation model that addresses the scarcity of reference answers in the target language by learning to assign scores to responses based on easily available reference answers in English. Our experiments demonstrate that Hercule aligns more closely with human judgments compared to proprietary models, demonstrating the effectiveness of such cross-lingual evaluation in low resource scenarios. Further, it is also effective in zero-shot evaluation on unseen languages. This study is the first comprehensive examination of cross-lingual evaluation using LLMs, presenting a scalable and effective approach for multilingual assessment. All code, datasets, and models will be publicly available to enable further research in this important area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13919v1">LLM Agent Honeypot: Monitoring AI Hacking Agents in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      We introduce the LLM Honeypot, a system for monitoring autonomous AI hacking agents. We deployed a customized SSH honeypot and applied prompt injections with temporal analysis to identify LLM-based agents among attackers. Over a trial run of a few weeks in a public environment, we collected 800,000 hacking attempts and 6 potential AI agents, which we plan to analyze in depth in future work. Our objectives aim to improve awareness of AI hacking agents and enhance preparedness for their risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12532v2">MedAide: Towards an Omni Medical Aide via Specialized LLM-based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 LLM-based Multi-Agent Collaboration for Medical Applications
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-driven interactive systems currently show potential promise in healthcare domains. Despite their remarkable capabilities, LLMs typically lack personalized recommendations and diagnosis analysis in sophisticated medical applications, causing hallucinations and performance bottlenecks. To address these challenges, this paper proposes MedAide, an LLM-based omni medical multi-agent collaboration framework for specialized healthcare services. Specifically, MedAide first performs query rewriting through retrieval-augmented generation to accomplish accurate medical intent understanding. Immediately, we devise a contextual encoder to obtain intent prototype embeddings, which are used to recognize fine-grained intents by similarity matching. According to the intent relevance, the activated agents collaborate effectively to provide integrated decision analysis. Extensive experiments are conducted on four medical benchmarks with composite intents. Experimental results from automated metrics and expert doctor evaluations show that MedAide outperforms current LLMs and improves their medical proficiency and strategic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12464v2">Enhancing LLM Trading Performance with Fact-Subjectivity Aware Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      While many studies prove more advanced LLMs perform better on tasks such as math and coding, we notice that in cryptocurrency trading, stronger LLMs work worse than weaker LLMs often. To study how this counter-intuitive phenomenon occurs, we examine the LLM reasoning processes on making trading decisions. We find that separating the reasoning process into factual and subjective components can lead to higher profits. Building on this insight, we introduce a multi-agent framework, FS-ReasoningAgent, which enables LLMs to recognize and learn from both factual and subjective reasoning. Extensive experiments demonstrate that this framework enhances LLM trading performance in cryptocurrency markets. Additionally, an ablation study reveals that relying on subjective news tends to generate higher returns in bull markets, whereas focusing on factual information yields better results in bear markets. Our code and data are available at \url{https://anonymous.4open.science/r/FS-ReasoningAgent-B55F/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11643v2">Combating Phone Scams with LLM-based Detection: Where Do We Stand?</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 2 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Phone scams pose a significant threat to individuals and communities, causing substantial financial losses and emotional distress. Despite ongoing efforts to combat these scams, scammers continue to adapt and refine their tactics, making it imperative to explore innovative countermeasures. This research explores the potential of large language models (LLMs) to provide detection of fraudulent phone calls. By analyzing the conversational dynamics between scammers and victims, LLM-based detectors can identify potential scams as they occur, offering immediate protection to users. While such approaches demonstrate promising results, we also acknowledge the challenges of biased datasets, relatively low recall, and hallucinations that must be addressed for further advancement in this field
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13343v1">Do LLMs Overcome Shortcut Learning? An Evaluation of Shortcut Challenges in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in various natural language processing tasks. However, LLMs may rely on dataset biases as shortcuts for prediction, which can significantly impair their robustness and generalization capabilities. This paper presents Shortcut Suite, a comprehensive test suite designed to evaluate the impact of shortcuts on LLMs' performance, incorporating six shortcut types, five evaluation metrics, and four prompting strategies. Our extensive experiments yield several key findings: 1) LLMs demonstrate varying reliance on shortcuts for downstream tasks, significantly impairing their performance. 2) Larger LLMs are more likely to utilize shortcuts under zero-shot and few-shot in-context learning prompts. 3) Chain-of-thought prompting notably reduces shortcut reliance and outperforms other prompting strategies, while few-shot prompts generally underperform compared to zero-shot prompts. 4) LLMs often exhibit overconfidence in their predictions, especially when dealing with datasets that contain shortcuts. 5) LLMs generally have a lower explanation quality in shortcut-laden datasets, with errors falling into three types: distraction, disguised comprehension, and logical fallacy. Our findings offer new insights for evaluating robustness and generalization in LLMs and suggest potential directions for mitigating the reliance on shortcuts. The code is available at \url {https://github.com/yyhappier/ShortcutSuite.git}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13341v1">Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      High quality annotations are increasingly a bottleneck in the explosively growing machine learning ecosystem. Scalable evaluation methods that avoid costly annotation have therefore become an important research ambition. Many hope to use strong existing models in lieu of costly labels to provide cheap model evaluations. Unfortunately, this method of using models as judges introduces biases, such as self-preferencing, that can distort model comparisons. An emerging family of debiasing tools promises to fix these issues by using a few high quality labels to debias a large number of model judgments. In this paper, we study how far such debiasing methods, in principle, can go. Our main result shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half. Our result speaks to the severe limitations of the LLM-as-a-judge paradigm at the evaluation frontier where the goal is to assess newly released models that are possibly better than the judge. Through an empirical evaluation, we demonstrate that the sample size savings achievable in practice are even more modest than what our theoretical limit suggests. Along the way, our work provides new observations about debiasing methods for model evaluation, and points out promising avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13326v1">Comparing the Utility, Preference, and Performance of Course Material Search Functionality and Retrieval-Augmented Generation Large Language Model (RAG-LLM) AI Chatbots in Information-Seeking Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Providing sufficient support for students requires substantial resources, especially considering the growing enrollment numbers. Students need help in a variety of tasks, ranging from information-seeking to requiring support with course assignments. To explore the utility of recent large language models (LLMs) as a support mechanism, we developed an LLM-powered AI chatbot that augments the answers that are produced with information from the course materials. To study the effect of the LLM-powered AI chatbot, we conducted a lab-based user study (N=14), in which the participants worked on tasks from a web software development course. The participants were divided into two groups, where one of the groups first had access to the chatbot and then to a more traditional search functionality, while another group started with the search functionality and was then given the chatbot. We assessed the participants' performance and perceptions towards the chatbot and the search functionality and explored their preferences towards the support functionalities. Our findings highlight that both support mechanisms are seen as useful and that support mechanisms work well for specific tasks, while less so for other tasks. We also observe that students tended to prefer the second support mechanism more, where students who were first given the chatbot tended to prefer the search functionality and vice versa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11522v2">Leveraging LLM Embeddings for Cross Dataset Label Alignment and Zero Shot Music Emotion Prediction</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      In this work, we present a novel method for music emotion recognition that leverages Large Language Model (LLM) embeddings for label alignment across multiple datasets and zero-shot prediction on novel categories. First, we compute LLM embeddings for emotion labels and apply non-parametric clustering to group similar labels, across multiple datasets containing disjoint labels. We use these cluster centers to map music features (MERT) to the LLM embedding space. To further enhance the model, we introduce an alignment regularization that enables dissociation of MERT embeddings from different clusters. This further enhances the model's ability to better adaptation to unseen datasets. We demonstrate the effectiveness of our approach by performing zero-shot inference on a new dataset, showcasing its ability to generalize to unseen labels without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04168v3">Perceive, Reflect, and Plan: Designing LLM Agent for Goal-Directed City Navigation without Instructions</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      This paper considers a scenario in city navigation: an AI agent is provided with language descriptions of the goal location with respect to some well-known landmarks; By only observing the scene around, including recognizing landmarks and road network connections, the agent has to make decisions to navigate to the goal location without instructions. This problem is very challenging, because it requires agent to establish self-position and acquire spatial representation of complex urban environment, where landmarks are often invisible. In the absence of navigation instructions, such abilities are vital for the agent to make high-quality decisions in long-range city navigation. With the emergent reasoning ability of large language models (LLMs), a tempting baseline is to prompt LLMs to "react" on each observation and make decisions accordingly. However, this baseline has very poor performance that the agent often repeatedly visits same locations and make short-sighted, inconsistent decisions. To address these issues, this paper introduces a novel agentic workflow featured by its abilities to perceive, reflect and plan. Specifically, we find LLaVA-7B can be fine-tuned to perceive the direction and distance of landmarks with sufficient accuracy for city navigation. Moreover, reflection is achieved through a memory mechanism, where past experiences are stored and can be retrieved with current perception for effective decision argumentation. Planning uses reflection results to produce long-term plans, which can avoid short-sighted decisions in long-range navigation. We show the designed workflow significantly improves navigation ability of the LLM agent compared with the state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12187v2">DAQ: Density-Aware Post-Training Weight-Only Quantization For LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in various tasks but face deployment challenges due to hardware constraints. We propose density-aware post-training weight-only quantization (DAQ), which has two stages: 1) density-centric alignment, which identifies the center of high-density weights and centers the dynamic range on this point to align high-density weight regions with floating-point high-precision regions; 2) learnable dynamic range adjustment, which adjusts the dynamic range by optimizing quantization parameters (i.e., scale and zero-point) based on the impact of weights on the model output. Experiments on LLaMA and LLaMA-2 show that DAQ consistently outperforms the best baseline method, reducing perplexity loss by an average of 22.8% on LLaMA and 19.6% on LLaMA-2. Our code is available at https://github.com/LuoYingSong/DAQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13246v1">Atomic Calibration of LLMs in Long-Form Generations</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often suffer from hallucinations, posing significant challenges for real-world applications. Confidence calibration, which estimates the underlying uncertainty of model predictions, is essential to enhance the LLMs' trustworthiness. Existing research on LLM calibration has primarily focused on short-form tasks, providing a single confidence score at the response level (macro calibration). However, this approach is insufficient for long-form generations, where responses often contain more complex statements and may include both accurate and inaccurate information. Therefore, we introduce atomic calibration, a novel approach that evaluates factuality calibration at a fine-grained level by breaking down long responses into atomic claims. We classify confidence elicitation methods into discriminative and generative types and demonstrate that their combination can enhance calibration. Our extensive experiments on various LLMs and datasets show that atomic calibration is well-suited for long-form generation and can also improve macro calibration results. Additionally, atomic calibration reveals insightful patterns in LLM confidence throughout the generation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17484v3">MedCare: Advancing Medical LLMs through Decoupling Clinical Alignment and Knowledge Aggregation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 EMNLP2024 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown substantial progress in natural language understanding and generation, proving valuable especially in the medical field. Despite advancements, challenges persist due to the complexity and diversity inherent in medical tasks, which can be categorized as knowledge-intensive tasks and alignment-required tasks. Previous approaches either ignore the latter task or focus on a minority of tasks and hence lose generalization. To address these drawbacks, we propose a progressive fine-tuning pipeline. This pipeline employs a Knowledge Aggregator and a Noise aggregator to encode diverse knowledge in the first stage and filter out detrimental information. In the second stage, we drop the Noise Aggregator to avoid the interference of suboptimal representation and leverage an additional alignment module optimized towards an orthogonal direction to the knowledge space to mitigate knowledge forgetting. Based on this two-stage paradigm, we proposed a Medical LLM through decoupling Clinical Alignment and Knowledge Aggregation (MedCare), which is designed to achieve state-of-the-art (SOTA) performance on over 20 medical tasks, as well as SOTA results on specific medical alignment tasks. Various model sizes of MedCare (1.8B, 7B, 14B) all demonstrate significant improvements over existing models with similar model sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13210v1">FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Summarization is one of the most common tasks performed by large language models (LLMs), especially in applications like Retrieval-Augmented Generation (RAG). However, existing evaluations of hallucinations in LLM-generated summaries, and evaluations of hallucination detection models both suffer from a lack of diversity and recency in the LLM and LLM families considered. This paper introduces FaithBench, a summarization hallucination benchmark comprising challenging hallucinations made by 10 modern LLMs from 8 different families, with ground truth annotations by human experts. ``Challenging'' here means summaries on which popular, state-of-the-art hallucination detection models, including GPT-4o-as-a-judge, disagreed on. Our results show GPT-4o and GPT-3.5-Turbo produce the least hallucinations. However, even the best hallucination detection models have near 50\% accuracies on FaithBench, indicating lots of room for future improvement. The repo is https://github.com/vectara/FaithBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19598v2">Mixture of In-Context Experts Enhance LLMs' Long Context Awareness</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted by Neurips2024
    </div>
    <details class="paper-abstract">
      Many studies have revealed that large language models (LLMs) exhibit uneven awareness of different contextual positions. Their limited context awareness can lead to overlooking critical information and subsequent task failures. While several approaches have been proposed to enhance LLMs' context awareness, achieving both effectiveness and efficiency remains challenging. In this paper, for LLMs utilizing RoPE as position embeddings, we introduce a novel method called "Mixture of In-Context Experts" (MoICE) to address this challenge. MoICE comprises two key components: a router integrated into each attention head within LLMs and a lightweight router-only training optimization strategy: (1) MoICE views each RoPE angle as an `in-context' expert, demonstrated to be capable of directing the attention of a head to specific contextual positions. Consequently, each attention head flexibly processes tokens using multiple RoPE angles dynamically selected by the router to attend to the needed positions. This approach mitigates the risk of overlooking essential contextual information. (2) The router-only training strategy entails freezing LLM parameters and exclusively updating routers for only a few steps. When applied to open-source LLMs including Llama and Mistral, MoICE surpasses prior methods across multiple tasks on long context understanding and generation, all while maintaining commendable inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12424v5">Tables as Texts or Images: Evaluating the Table Reasoning Ability of LLMs and MLLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Accepted to ACL 2024 Findings; Naihao and Zhenjie contributed equally to the project; Data available at: https://github.com/dnaihao/Tables-as-Texts-or-Images
    </div>
    <details class="paper-abstract">
      In this paper, we investigate the effectiveness of various LLMs in interpreting tabular data through different prompting strategies and data formats. Our analyses extend across six benchmarks for table-related tasks such as question-answering and fact-checking. We introduce for the first time the assessment of LLMs' performance on image-based table representations. Specifically, we compare five text-based and three image-based table representations, demonstrating the role of representation and prompting on LLM performance. Our study provides insights into the effective use of LLMs on table-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13155v1">SLM-Mod: Small Language Models Surpass LLMs at Content Moderation</a></div>
    <div class="paper-meta">
      📅 2024-10-17
      | 💬 Preprint: 15 pages, 8 figures, 8 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in many natural language understanding tasks, including content moderation. However, these models can be expensive to query in real-time and do not allow for a community-specific approach to content moderation. To address these challenges, we explore the use of open-source small language models (SLMs) for community-specific content moderation tasks. We fine-tune and evaluate SLMs (less than 15B parameters) by comparing their performance against much larger open- and closed-sourced models. Using 150K comments from 15 popular Reddit communities, we find that SLMs outperform LLMs at content moderation -- 11.5% higher accuracy and 25.7% higher recall on average across all communities. We further show the promise of cross-community content moderation, which has implications for new communities and the development of cross-platform moderation techniques. Finally, we outline directions for future work on language model based content moderation. Code and links to HuggingFace models can be found at https://github.com/AGoyal0512/SLM-Mod.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.12052v1">Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction</a></div>
    <div class="paper-meta">
      📅 2024-10-17
    </div>
    <details class="paper-abstract">
      Accurate skill extraction from job descriptions is crucial in the hiring process but remains challenging. Named Entity Recognition (NER) is a common approach used to address this issue. With the demonstrated success of large language models (LLMs) in various NLP tasks, including NER, we propose fine-tuning a specialized Skill-LLM and a light weight model to improve the precision and quality of skill extraction. In our study, we evaluated the fine-tuned Skill-LLM and the light weight model using a benchmark dataset and compared its performance against state-of-the-art (SOTA) methods. Our results show that this approach outperforms existing SOTA techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06682v3">Self-Reflection in LLM Agents: Effects on Problem-Solving Performance</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      In this study, we investigated the effects of self-reflection in large language models (LLMs) on problem-solving performance. We instructed nine popular LLMs to answer a series of multiple-choice questions to provide a performance baseline. For each incorrectly answered question, we instructed eight types of self-reflecting LLM agents to reflect on their mistakes and provide themselves with guidance to improve problem-solving. Then, using this guidance, each self-reflecting agent attempted to re-answer the same questions. Our results indicate that LLM agents are able to significantly improve their problem-solving performance through self-reflection ($p < 0.001$). In addition, we compared the various types of self-reflection to determine their individual contribution to performance. All code and data are available on GitHub at https://github.com/matthewrenze/self-reflection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13032v1">Hypothesis Testing the Circuit Hypothesis in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Code available here: https://github.com/blei-lab/circuitry
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate surprising capabilities, but we do not understand how they are implemented. One hypothesis suggests that these capabilities are primarily executed by small subnetworks within the LLM, known as circuits. But how can we evaluate this hypothesis? In this paper, we formalize a set of criteria that a circuit is hypothesized to meet and develop a suite of hypothesis tests to evaluate how well circuits satisfy them. The criteria focus on the extent to which the LLM's behavior is preserved, the degree of localization of this behavior, and whether the circuit is minimal. We apply these tests to six circuits described in the research literature. We find that synthetic circuits -- circuits that are hard-coded in the model -- align with the idealized properties. Circuits discovered in Transformer models satisfy the criteria to varying degrees. To facilitate future empirical studies of circuits, we created the \textit{circuitry} package, a wrapper around the \textit{TransformerLens} library, which abstracts away lower-level manipulations of hooks and activations. The software is available at \url{https://github.com/blei-lab/circuitry}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00782v1">TradExpert: Revolutionizing Trading with Mixture of Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The integration of Artificial Intelligence (AI) in the financial domain has opened new avenues for quantitative trading, particularly through the use of Large Language Models (LLMs). However, the challenge of effectively synthesizing insights from diverse data sources and integrating both structured and unstructured data persists. This paper presents TradeExpert, a novel framework that employs a mix of experts (MoE) approach, using four specialized LLMs, each analyzing distinct sources of financial data, including news articles, market data, alpha factors, and fundamental data. The insights of these expert LLMs are further synthesized by a General Expert LLM to make a final prediction or decision. With specific prompts, TradeExpert can be switched between the prediction mode and the ranking mode for stock movement prediction and quantitative stock trading, respectively. In addition to existing benchmarks, we also release a large-scale financial dataset to comprehensively evaluate TradeExpert's effectiveness. Our experimental results demonstrate TradeExpert's superior performance across all trading scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13007v1">Codellm-Devkit: A Framework for Contextualizing Code LLMs with Program Analysis Insights</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models for Code (or code LLMs) are increasingly gaining popularity and capabilities, offering a wide array of functionalities such as code completion, code generation, code summarization, test generation, code translation, and more. To leverage code LLMs to their full potential, developers must provide code-specific contextual information to the models. These are typically derived and distilled using program analysis tools. However, there exists a significant gap--these static analysis tools are often language-specific and come with a steep learning curve, making their effective use challenging. These tools are tailored to specific program languages, requiring developers to learn and manage multiple tools to cover various aspects of the their code base. Moreover, the complexity of configuring and integrating these tools into the existing development environments add an additional layer of difficulty. This challenge limits the potential benefits that could be gained from more widespread and effective use of static analysis in conjunction with LLMs. To address this challenge, we present codellm-devkit (hereafter, `CLDK'), an open-source library that significantly simplifies the process of performing program analysis at various levels of granularity for different programming languages to support code LLM use cases. As a Python library, CLDK offers developers an intuitive and user-friendly interface, making it incredibly easy to provide rich program analysis context to code LLMs. With this library, developers can effortlessly integrate detailed, code-specific insights that enhance the operational efficiency and effectiveness of LLMs in coding tasks. CLDK is available as an open-source library at https://github.com/IBM/codellm-devkit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12985v1">Leveraging LLMs for Translating and Classifying Mental Health Data</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in medical fields. In mental health support, the early identification of linguistic markers associated with mental health conditions can provide valuable support to mental health professionals, and reduce long waiting times for patients. Despite the benefits of LLMs for mental health support, there is limited research on their application in mental health systems for languages other than English. Our study addresses this gap by focusing on the detection of depression severity in Greek through user-generated posts which are automatically translated from English. Our results show that GPT3.5-turbo is not very successful in identifying the severity of depression in English, and it has a varying performance in Greek as well. Our study underscores the necessity for further research, especially in languages with less resources. Also, careful implementation is necessary to ensure that LLMs are used effectively in mental health platforms, and human supervision remains crucial to avoid misdiagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19209v2">VideoTree: Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 23 pages, first three authors contributed equally; Project page: https://videotree2024.github.io/
    </div>
    <details class="paper-abstract">
      Long-form video understanding has been a challenging task due to the high redundancy in video data and the abundance of query-irrelevant information. To tackle this challenge, we propose VideoTree, a training-free framework which builds a query-adaptive and hierarchical video representation for LLM reasoning over long-form videos. First, VideoTree extracts query-relevant information from the input video through an iterative process, progressively refining the selection of keyframes based on their relevance to the query. Furthermore, VideoTree leverages the inherent hierarchical structure of long video data, which is often overlooked by existing LLM-based methods. Specifically, we incorporate multigranularity information into a tree-based representation, allowing VideoTree to extract query-relevant details from long videos in a coarse-to-fine manner. This enables the model to effectively handle a wide range of video queries with varying levels of detail. Finally, VideoTree aggregates the hierarchical query-relevant information within the tree structure and feeds it into an LLM reasoning model to answer the query. Our experiments show that our training-free method improves both reasoning accuracy and efficiency compared to existing methods. Specifically, VideoTree outperforms the existing training-free approaches on the popular EgoSchema and NExT-QA benchmarks with less inference time, achieving 61.1% and 75.6% accuracy on the test set without additional video-specific training. Moreover, on the long split of Video-MME benchmark (average 44 minutes), the training-free VideoTree framework achieves better performance than the strong proprietary GPT-4V model and other MLLMs that were extensively trained on video data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12952v1">Facilitating Multi-turn Function Calling for LLMs via Compositional Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited significant potential in performing diverse tasks, including the ability to call functions or use external tools to enhance their performance. While current research on function calling by LLMs primarily focuses on single-turn interactions, this paper addresses the overlooked necessity for LLMs to engage in multi-turn function calling--critical for handling compositional, real-world queries that require planning with functions but not only use functions. To facilitate this, we introduce an approach, BUTTON, which generates synthetic compositional instruction tuning data via bottom-up instruction construction and top-down trajectory generation. In the bottom-up phase, we generate simple atomic tasks based on real-world scenarios and build compositional tasks using heuristic strategies based on atomic tasks. Corresponding functions are then developed for these compositional tasks. The top-down phase features a multi-agent environment where interactions among simulated humans, assistants, and tools are utilized to gather multi-turn function calling trajectories. This approach ensures task compositionality and allows for effective function and trajectory generation by examining atomic tasks within compositional tasks. We produce a dataset BUTTONInstruct comprising 8k data points and demonstrate its effectiveness through extensive experiments across various LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12934v1">Enhancing Mathematical Reasoning in LLMs by Stepwise Correction</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Best-of-N decoding methods instruct large language models (LLMs) to generate multiple solutions, score each using a scoring function, and select the highest scored as the final answer to mathematical reasoning problems. However, this repeated independent process often leads to the same mistakes, making the selected solution still incorrect. We propose a novel prompting method named Stepwise Correction (StepCo) that helps LLMs identify and revise incorrect steps in their generated reasoning paths. It iterates verification and revision phases that employ a process-supervised verifier. The verify-then-revise process not only improves answer correctness but also reduces token consumption with fewer paths needed to generate. With StepCo, a series of LLMs demonstrate exceptional performance. Notably, using GPT-4o as the backend LLM, StepCo achieves an average accuracy of 94.1 across eight datasets, significantly outperforming the state-of-the-art Best-of-N method by +2.4, while reducing token consumption by 77.8%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12924v1">Interpreting token compositionality in LLMs: A robustness analysis</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 15 pages, 2 Figures, 7 tables
    </div>
    <details class="paper-abstract">
      Understanding the internal mechanisms of large language models (LLMs) is integral to enhancing their reliability, interpretability, and inference processes. We present Constituent-Aware Pooling (CAP), a methodology designed to analyse how LLMs process compositional linguistic structures. Grounded in principles of compositionality, mechanistic interpretability, and information gain theory, CAP systematically intervenes in model activations through constituent-based pooling at various model levels. Our experiments on inverse definition modelling, hypernym and synonym prediction reveal critical insights into transformers' limitations in handling compositional abstractions. No specific layer integrates tokens into unified semantic representations based on their constituent parts. We observe fragmented information processing, which intensifies with model size, suggesting that larger models struggle more with these interventions and exhibit greater information dispersion. This fragmentation likely stems from transformers' training objectives and architectural design, preventing systematic and cohesive representations. Our findings highlight fundamental limitations in current transformer architectures regarding compositional semantics processing and model interpretability, underscoring the critical need for novel approaches in LLM design to address these challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12784v1">JudgeBench: A Benchmark for Evaluating LLM-based Judges</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      LLM-based judges have emerged as a scalable alternative to human evaluation and are increasingly used to assess, compare, and improve models. However, the reliability of LLM-based judges themselves is rarely scrutinized. As LLMs become more advanced, their responses grow more sophisticated, requiring stronger judges to evaluate them. Existing benchmarks primarily focus on a judge's alignment with human preferences, but often fail to account for more challenging tasks where crowdsourced human preference is a poor indicator of factual and logical correctness. To address this, we propose a novel evaluation framework to objectively evaluate LLM-based judges. Based on this framework, we propose JudgeBench, a benchmark for evaluating LLM-based judges on challenging response pairs spanning knowledge, reasoning, math, and coding. JudgeBench leverages a novel pipeline for converting existing difficult datasets into challenging response pairs with preference labels reflecting objective correctness. Our comprehensive evaluation on a collection of prompted judges, fine-tuned judges, multi-agent judges, and reward models shows that JudgeBench poses a significantly greater challenge than previous benchmarks, with many strong models (e.g., GPT-4o) performing just slightly better than random guessing. Overall, JudgeBench offers a reliable platform for assessing increasingly advanced LLM-based judges. Data and code are available at https://github.com/ScalerLab/JudgeBench .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12782v1">In-Context Learning Enables Robot Action Prediction in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have achieved remarkable success using in-context learning (ICL) in the language domain. However, leveraging the ICL capabilities within LLMs to directly predict robot actions remains largely unexplored. In this paper, we introduce RoboPrompt, a framework that enables off-the-shelf text-only LLMs to directly predict robot actions through ICL without training. Our approach first heuristically identifies keyframes that capture important moments from an episode. Next, we extract end-effector actions from these keyframes as well as the estimated initial object poses, and both are converted into textual descriptions. Finally, we construct a structured template to form ICL demonstrations from these textual descriptions and a task instruction. This enables an LLM to directly predict robot actions at test time. Through extensive experiments and analysis, RoboPrompt shows stronger performance over zero-shot and ICL baselines in simulated and real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10759v2">SplitLLM: Collaborative Inference of LLMs for Model Placement and Throughput Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been a disruptive innovation in recent years, and they play a crucial role in our daily lives due to their ability to understand and generate human-like text. Their capabilities include natural language understanding, information retrieval and search, translation, chatbots, virtual assistance, and many more. However, it is well known that LLMs are massive in terms of the number of parameters. Additionally, the self-attention mechanism in the underlying architecture of LLMs, Transformers, has quadratic complexity in terms of both computation and memory with respect to the input sequence length. For these reasons, LLM inference is resource-intensive, and thus, the throughput of LLM inference is limited, especially for the longer sequences. In this report, we design a collaborative inference architecture between a server and its clients to alleviate the throughput limit. In this design, we consider the available resources on both sides, i.e., the computation and communication costs. We develop a dynamic programming-based algorithm to optimally allocate computation between the server and the client device to increase the server throughput, while not violating the service level agreement (SLA). We show in the experiments that we are able to efficiently distribute the workload allowing for roughly 1/3 reduction in the server workload, while achieving 19 percent improvement over a greedy method. As a result, we are able to demonstrate that, in an environment with different types of LLM inference requests, the throughput of the server is improved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12707v1">FusionLLM: A Decentralized LLM Training System on Geo-distributed GPUs with Adaptive Compression</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      To alleviate hardware scarcity in training large deep neural networks (DNNs), particularly large language models (LLMs), we present FusionLLM, a decentralized training system designed and implemented for training DNNs using geo-distributed GPUs across different computing clusters or individual devices. Decentralized training faces significant challenges regarding system design and efficiency, including: 1) the need for remote automatic differentiation (RAD), 2) support for flexible model definitions and heterogeneous software, 3) heterogeneous hardware leading to low resource utilization or the straggler problem, and 4) slow network communication. To address these challenges, in the system design, we represent the model as a directed acyclic graph of operators (OP-DAG). Each node in the DAG represents the operator in the DNNs, while the edge represents the data dependency between operators. Based on this design, 1) users are allowed to customize any DNN without caring low-level operator implementation; 2) we enable the task scheduling with the more fine-grained sub-tasks, offering more optimization space; 3) a DAG runtime executor can implement RAD withour requiring the consistent low-level ML framework versions. To enhance system efficiency, we implement a workload estimator and design an OP-Fence scheduler to cluster devices with similar bandwidths together and partition the DAG to increase throughput. Additionally, we propose an AdaTopK compressor to adaptively compress intermediate activations and gradients at the slowest communication links. To evaluate the convergence and efficiency of our system and algorithms, we train ResNet-101 and GPT-2 on three real-world testbeds using 48 GPUs connected with 8 Mbps~10 Gbps networks. Experimental results demonstrate that our system and method can achieve 1.45 - 9.39x speedup compared to baseline methods while ensuring convergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15219v2">Unsupervised End-to-End Task-Oriented Dialogue with LLMs: The Power of the Noisy Channel</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 To be presented at Empirical Methods in Natural Language Processing (EMNLP 2024). 18 Pages, 8 Figures
    </div>
    <details class="paper-abstract">
      Training task-oriented dialogue systems typically requires turn-level annotations for interacting with their APIs: e.g. a dialogue state and the system actions taken at each step. These annotations can be costly to produce, error-prone, and require both domain and annotation expertise. With advances in LLMs, we hypothesize that unlabeled data and a schema definition are sufficient for building a working task-oriented dialogue system, completely unsupervised. We consider a novel unsupervised setting of only (1) a well-defined API schema (2) a set of unlabeled dialogues between a user and agent. We propose an innovative approach using expectation-maximization (EM) that infers turn-level annotations as latent variables using a noisy channel model to build an end-to-end dialogue agent. Evaluating our approach on the MultiWOZ benchmark, our method more than doubles the dialogue success rate of a strong GPT-3.5 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11167v2">ToBlend: Token-Level Blending With an Ensemble of LLMs to Attack AI-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Submitted to ARR Oct-2024 Cycle
    </div>
    <details class="paper-abstract">
      The robustness of AI-content detection models against sophisticated adversarial strategies, such as paraphrasing or word switching, is a rising concern in natural language generation (NLG) applications. This study proposes ToBlend, a novel token-level ensemble text generation method to challenge the robustness of current AI-content detection approaches by utilizing multiple sets of candidate generative large language models (LLMs). By randomly sampling token(s) from candidate LLMs sets, we find ToBlend significantly drops the performance of most mainstream AI-content detection methods. We evaluate the text quality produced under different ToBlend settings based on annotations from experienced human experts. We proposed a fine-tuned Llama3.1 model to distinguish the ToBlend generated text more accurately. Our findings underscore our proposed text generation approach's great potential in deceiving and improving detection models. Our datasets, codes, and annotations are open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15360v3">Reward-Robust RLHF in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to progress toward more advanced forms of intelligence, Reinforcement Learning from Human Feedback (RLHF) is increasingly seen as a key pathway toward achieving Artificial General Intelligence (AGI). However, the reliance on reward-model-based (RM-based) alignment methods introduces significant challenges due to the inherent instability and imperfections of Reward Models (RMs), which can lead to critical issues such as reward hacking and misalignment with human intentions. In this paper, we introduce a reward-robust RLHF framework aimed at addressing these fundamental challenges, paving the way for more reliable and resilient learning in LLMs. Our approach introduces a novel optimization objective that carefully balances performance and robustness by incorporating Bayesian Reward Model Ensembles (BRME) to model the uncertainty set of reward functions. This allows the framework to integrate both nominal performance and minimum reward signals, ensuring more stable learning even with imperfect RMs. Empirical results demonstrate that our framework consistently outperforms baselines across diverse benchmarks, showing improved accuracy and long-term stability. We also provide a theoretical analysis, demonstrating that reward-robust RLHF approaches the stability of constant reward settings, which proves to be acceptable even in a stochastic-case analysis. Together, these contributions highlight the framework potential to enhance both the performance and stability of LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07140v4">Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant success in reasoning tasks, including mathematical reasoning and logical deduction. Among these reasoning tasks, graph problems stand out due to their complexity and unique structural characteristics, attracting considerable attention from researchers. Previous studies have explored LLMs' graph reasoning abilities through various techniques, such as different encoding methods for graph structures and the use of carefully designed prompts. However, a critical factor has been mostly overlooked: the prompt sequential order in which graph descriptions are presented to the models. In this study, we present the first comprehensive analysis of how the order of graph descriptions impacts LLM performance. Specifically, we comprehensively evaluate four graph description orders across six graph problems using six mainstream LLMs. The results reveal that: (1) ordered graph descriptions significantly improve LLMs' comprehension of graph structures; (2) the robustness of LLMs to graph description order varies across different tasks; and (3) the impact of graph order on performance is closely related to the inherent characteristics of tasks. This study provides a critical advancement in the application of LLMs for solving graph-related problems, paving the way for future research to optimize model performance through strategic graph description ordering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12601v1">CCSBench: Evaluating Compositional Controllability in LLMs for Scientific Document Summarization</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      To broaden the dissemination of scientific knowledge to diverse audiences, scientific document summarization must simultaneously control multiple attributes such as length and empirical focus. However, existing research typically focuses on controlling single attributes, leaving the compositional control of multiple attributes underexplored. To address this gap, we introduce CCSBench, a benchmark for compositional controllable summarization in the scientific domain. Our benchmark enables fine-grained control over both explicit attributes (e.g., length), which are objective and straightforward, and implicit attributes (e.g., empirical focus), which are more subjective and conceptual. We conduct extensive experiments on GPT-4, LLaMA2, and other popular LLMs under various settings. Our findings reveal significant limitations in large language models' ability to balance trade-offs between control attributes, especially implicit ones that require deeper understanding and abstract reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12600v1">On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Evidence-enhanced detectors present remarkable abilities in identifying malicious social text with related evidence. However, the rise of large language models (LLMs) brings potential risks of evidence pollution to confuse detectors. This paper explores how to manipulate evidence, simulating potential misuse scenarios including basic pollution, and rephrasing or generating evidence by LLMs. To mitigate its negative impact, we propose three defense strategies from both the data and model sides, including machine-generated text detection, a mixture of experts, and parameter updating. Extensive experiments on four malicious social text detection tasks with ten datasets present that evidence pollution, especially the generate strategy, significantly compromises existing detectors. On the other hand, the defense strategies could mitigate evidence pollution, but they faced limitations for practical employment, such as the need for annotated data and huge inference costs. Further analysis illustrates that polluted evidence is of high quality, would compromise the model calibration, and could ensemble to amplify the negative impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12583v1">STRUX: An LLM for Decision-Making with Structured Explanations</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 10 pages, 7 figures, submitted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Countless decisions shape our daily lives, and it is paramount to understand the how and why behind these choices. In this paper, we introduce a new LLM decision-making framework called STRUX, which enhances LLM decision-making by providing structured explanations. These include favorable and adverse facts related to the decision, along with their respective strengths. STRUX begins by distilling lengthy information into a concise table of key facts. It then employs a series of self-reflection steps to determine which of these facts are pivotal, categorizing them as either favorable or adverse in relation to a specific decision. Lastly, we fine-tune an LLM to identify and prioritize these key facts to optimize decision-making. STRUX has been evaluated on the challenging task of forecasting stock investment decisions based on earnings call transcripts and demonstrated superior performance against strong baselines. It enhances decision transparency by allowing users to understand the impact of different factors, representing a meaningful step towards practical decision-making with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12319v3">The Comparative Trap: Pairwise Comparisons Amplifies Biased Preferences of LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used as evaluators for natural language generation tasks, ensuring unbiased assessments is essential. However, LLM evaluators often display biased preferences, such as favoring verbosity and authoritative tones. Our empirical analysis reveals that these biases are exacerbated in pairwise evaluation, where LLMs directly compare two outputs and easily prioritize superficial attributes. In contrast, pointwise evaluation, which assesses outputs independently, is less susceptible to such bias because each output is judged in isolation. To address the limitations of the pairwise evaluation, we introduce a novel evaluation method, PRePair, which integrates pointwise reasoning within a pairwise framework. PRePair effectively alleviates biased preference, improving performance on the adversarial benchmark (LLMBar) while outperforming pointwise evaluation on the standard benchmark (MT-Bench).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00906v2">Generative AI and Perceptual Harms: Who's Suspected of using LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into a variety of writing tasks. While these tools can help people by generating ideas or producing higher quality work, like many other AI tools they may risk causing a variety of harms, disproportionately burdening historically marginalized groups. In this work, we introduce and evaluate perceptual harm, a term for the harm caused to users when others perceive or suspect them of using AI. We examined perceptual harms in three online experiments, each of which entailed human participants evaluating the profiles for fictional freelance writers. We asked participants whether they suspected the freelancers of using AI, the quality of their writing, and whether they should be hired. We found some support for perceptual harms against for certain demographic groups, but that perceptions of AI use negatively impacted writing evaluations and hiring outcomes across the board.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12519v1">RosePO: Aligning LLM-based Recommenders with Human Values</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing interest in leveraging Large Language Models (LLMs) for recommendation systems, which usually adapt a pre-trained LLM to the recommendation scenario through supervised fine-tuning (SFT). However, both the pre-training and SFT stages fail to explicitly model the comparative relationships of a user's preferences on different items. To construct a "helpful and harmless" LLM-based recommender, we propose a general framework -- Recommendation with smoothing personalized Preference Optimization (RosePO), which better aligns with customized human values during the post-training stage. Specifically, in addition to the input and chosen response that naturally align with SFT data, we design a rejected sampling strategy tailored for enhancing helpfulness, along with two strategies aimed at mitigating biases to promote harmlessness. To ensure robustness against uncertain labels present in automatically constructed preference data, we introduce a personalized smoothing factor predicted by a preference oracle into the optimization objective. Evaluation on three real-world datasets demonstrates the effectiveness of our method, showcasing not only improved recommendation performance but also mitigation of semantic hallucination and popularity bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12499v1">With a Grain of SALT: Are LLMs Fair Across Social Dimensions?</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      This paper presents an analysis of biases in open-source Large Language Models (LLMs) across various genders, religions, and races. We introduce a methodology for generating a bias detection dataset using seven bias triggers: General Debate, Positioned Debate, Career Advice, Story Generation, Problem-Solving, Cover-Letter Writing, and CV Generation. We use GPT-4o to generate a diverse set of prompts for each trigger across various genders, religious and racial groups. We evaluate models from Llama and Gemma family on the generated dataset. We anonymise the LLM-generated text associated with each group using GPT-4o-mini and do a pairwise comparison using GPT-4o-as-a-Judge. To quantify bias in the LLM-generated text we use the number of wins and losses in the pairwise comparison. Our analysis spans three languages, English, German, and Arabic to explore how language influences bias manifestation. Our findings reveal that LLMs exhibit strong polarization toward certain groups across each category, with a notable consistency observed across models. However, when switching languages, variations and anomalies emerge, often attributable to cultural cues and contextual differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.08688v4">The Fellowship of the LLMs: Multi-Agent Workflows for Synthetic Preference Optimization Dataset Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for generating synthetic Preference Optimization (PO) datasets using multi-agent workflows. We evaluate the effectiveness and potential of these workflows in automating and enhancing the dataset generation process. PO dataset generation requires two modules: (1) response evaluation, and (2) response generation. In the response evaluation module, the responses from Large Language Models (LLMs) are evaluated and ranked - a task typically carried out by human annotators that we automate using LLMs. We assess the response evaluation module in a 2 step process. In step 1, we assess LLMs as evaluators using three distinct prompting strategies. In step 2, we apply the winning prompting strategy to compare the performance of LLM-as-a-Judge, LLMs-as-a-Jury, and LLM Debate. Our evaluation shows that GPT-4o-as-a-Judge is more consistent across all datasets. For the response generation module, we use the identified LLM evaluator configuration and compare different configurations of the LLM Feedback Loop. We use the win rate to determine the best multi-agent configuration for generation. Experimenting with various configurations, we find that the LLM Feedback Loop, with Llama as the generator and Gemma as the reviewer, achieves a notable 71.8% and 73.8% win rate over single-agent Llama and Gemma, respectively. After identifying the best configurations for both modules, we generate our PO datasets using the above pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12491v1">Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse RL</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 80.40% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12481v1">SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The past years have seen Large Language Models (LLMs) strive not only as generative models but also as agents solving textual sequential decision-making tasks. When facing complex environments where their zero-shot abilities are insufficient, recent work showed online Reinforcement Learning (RL) could be used for the LLM agent to discover and learn efficient strategies interactively. However, most prior work sticks to on-policy algorithms, which greatly reduces the scope of methods such agents could use for both exploration and exploitation, such as experience replay and hindsight relabeling. Yet, such methods may be key for LLM learning agents, and in particular when designing autonomous intrinsically motivated agents sampling and pursuing their own goals (i.e. autotelic agents). This paper presents and studies an adaptation of Soft Actor-Critic and hindsight relabeling to LLM agents. Our method not only paves the path towards autotelic LLM agents that learn online but can also outperform on-policy methods in more classic multi-goal RL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12480v1">KcMF: A Knowledge-compliant Framework for Schema and Entity Matching with Fine-tuning-free LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Schema and entity matching tasks are crucial for data integration and management. While large language models (LLMs) have shown promising results in these tasks, they suffer from hallucinations and confusion about task instructions. In this paper, we present the Knowledge-Compliant Matching Framework (KcMF), an LLM-based approach that addresses these issues without the need for domain-specific fine-tuning. KcMF employs a pseudo-code-based task decomposition strategy to adopt task-specific natural language statements that guide LLM reasoning and reduce confusion. We also propose two mechanisms, Dataset as Knowledge (DaK) and Example as Knowledge (EaK), to build domain knowledge sets when unstructured domain knowledge is lacking. Additionally, we introduce a result-ensembling strategy to leverage multiple knowledge sources and suppress poorly formatted outputs. Comprehensive evaluations on schema and entity matching tasks demonstrate that KcMF outperforms previous non-LLM state-of-the-art (SOTA) methods by an average F1 score of 22.9% and competes effectively with SOTA fine-tuned LLMs. Moreover, KcMF generalizes well across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12470v1">Learning to Predict Usage Options of Product Reviews with LLM-Generated Labels</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Annotating large datasets can be challenging. However, crowd-sourcing is often expensive and can lack quality, especially for non-trivial tasks. We propose a method of using LLMs as few-shot learners for annotating data in a complex natural language task where we learn a standalone model to predict usage options for products from customer reviews. We also propose a new evaluation metric for this scenario, HAMS4, that can be used to compare a set of strings with multiple reference sets. Learning a custom model offers individual control over energy efficiency and privacy measures compared to using the LLM directly for the sequence-to-sequence task. We compare this data annotation approach with other traditional methods and demonstrate how LLMs can enable considerable cost savings. We find that the quality of the resulting data exceeds the level attained by third-party vendor services and that GPT-4-generated labels even reach the level of domain experts. We make the code and generated labels publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12445v1">Open Ko-LLM Leaderboard2: Bridging Foundational and Practical Evaluation for Korean LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The Open Ko-LLM Leaderboard has been instrumental in benchmarking Korean Large Language Models (LLMs), yet it has certain limitations. Notably, the disconnect between quantitative improvements on the overly academic leaderboard benchmarks and the qualitative impact of the models should be addressed. Furthermore, the benchmark suite is largely composed of translated versions of their English counterparts, which may not fully capture the intricacies of the Korean language. To address these issues, we propose Open Ko-LLM Leaderboard2, an improved version of the earlier Open Ko-LLM Leaderboard. The original benchmarks are entirely replaced with new tasks that are more closely aligned with real-world capabilities. Additionally, four new native Korean benchmarks are introduced to better reflect the distinct characteristics of the Korean language. Through these refinements, Open Ko-LLM Leaderboard2 seeks to provide a more meaningful evaluation for advancing Korean LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11507v2">Revisiting Benchmark and Assessment: An Agent-based Exploratory Dynamic Evaluation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      While various vertical domain large language models (LLMs) have been developed, the challenge of automatically evaluating their performance across different domains remains significant. Current benchmark-based evaluation methods exhibit rigid, aimless interactions and rely on pre-collected static datasets that are costly to build, inflexible across domains, and misaligned with practical user needs. To address this issue, we revisit the evaluation components and introduce two concepts: Benchmark+, which extends traditional question-answer benchmark into a more flexible "strategy-criterion" format; and Assessment+, which enhances the interaction process, enabling deeper exploration and supporting both quantitative metrics and qualitative insights. These concepts capture the nuanced behaviors of LLMs through richer, multi-turn interactions. We propose an agent-based evaluation framework called TestAgent, which implements these concepts through retrieval augmented generation and reinforcement learning. Experiments on tasks ranging from constructing vertical domain evaluation to activating existing benchmarks demonstrate the effectiveness of TestAgent across various scenarios. We believe this work offers an interesting perspective on automatic evaluation for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10441v2">Free Video-LLM: Prompt-guided Visual Perception for Efficient Training-free Video LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Tech report
    </div>
    <details class="paper-abstract">
      Vision-language large models have achieved remarkable success in various multi-modal tasks, yet applying them to video understanding remains challenging due to the inherent complexity and computational demands of video data. While training-based video-LLMs deliver high performance, they often require substantial resources for training and inference. Conversely, training-free approaches offer a more efficient alternative by adapting pre-trained image-LLMs models for video tasks without additional training, but they face inference efficiency bottlenecks due to the large number of visual tokens generated from video frames. In this work, we present a novel prompt-guided visual perception framework (abbreviated as Free Video-LLM) for efficient inference of training-free video LLMs. The proposed framework decouples spatial-temporal dimension and performs temporal frame sampling and spatial RoI cropping respectively based on task-specific prompts. Our method effectively reduces the number of visual tokens while maintaining high performance across multiple video question-answering benchmarks. Extensive experiments demonstrate that our approach achieves competitive results with significantly fewer tokens, offering an optimal trade-off between accuracy and computational efficiency compared to state-of-the-art video LLMs. The code will be available at https://github.com/contrastive/FreeVideoLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12405v1">ProSA: Assessing and Understanding the Prompt Sensitivity of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 EMNLP 2024, Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across various tasks, but their performance is highly sensitive to the prompts utilized. This variability poses challenges for accurate assessment and user satisfaction. Current research frequently overlooks instance-level prompt variations and their implications on subjective evaluations. To address these shortcomings, we introduce ProSA, a framework designed to evaluate and comprehend prompt sensitivity in LLMs. ProSA incorporates a novel sensitivity metric, PromptSensiScore, and leverages decoding confidence to elucidate underlying mechanisms. Our extensive study, spanning multiple tasks, uncovers that prompt sensitivity fluctuates across datasets and models, with larger models exhibiting enhanced robustness. We observe that few-shot examples can alleviate this sensitivity issue, and subjective evaluations are also susceptible to prompt sensitivities, particularly in complex, reasoning-oriented tasks. Furthermore, our findings indicate that higher model confidence correlates with increased prompt robustness. We believe this work will serve as a helpful tool in studying prompt sensitivity of LLMs. The project is released at: https://github.com/open-compass/ProSA .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.03003v3">Explore, Select, Derive, and Recall: Augmenting LLM with Human-like Memory for Mobile Task Automation</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has opened up new opportunities in the field of mobile task automation. Their superior language understanding and reasoning capabilities allow users to automate complex and repetitive tasks. However, due to the inherent unreliability and high operational cost of LLMs, their practical applicability is quite limited. To address these issues, this paper introduces MobileGPT, an innovative LLM-based mobile task automator equipped with a human-like app memory. MobileGPT emulates the cognitive process of humans interacting with a mobile app -- explore, select, derive, and recall. This approach allows for a more precise and efficient learning of a task's procedure by breaking it down into smaller, modular sub-tasks that can be re-used, re-arranged, and adapted for various objectives. We implement MobileGPT using online LLMs services (GPT-3.5 and GPT-4) and evaluate its performance on a dataset of 185 tasks across 18 mobile apps. The results indicate that MobileGPT can automate and learn new tasks with 82.7% accuracy, and is able to adapt them to different contexts with near perfect (98.75%) accuracy while reducing both latency and cost by 62.5% and 68.8%, respectively, compared to the GPT-4 powered baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13903v1">CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Proprietary large language models (LLMs) demonstrate exceptional generalization ability across various tasks. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security threats: attackers who obtain an edge-deployed LLM can easily use it as a base model for various tasks due to its high generalization ability, which we call foundational capability stealing. Unfortunately, existing model protection mechanisms are often task-specific and fail to protect general-purpose LLMs, as they mainly focus on protecting task-related parameters using trusted execution environments (TEEs). Although some recent TEE-based methods are able to protect the overall model parameters in a computation-efficient way, they still suffer from prohibitive communication costs between TEE and CPU/GPU, making it impractical to deploy for edge LLMs. To protect the foundational capabilities of edge LLMs, we propose CoreGuard, a computation- and communication-efficient model protection approach against model stealing on edge devices. The core component of CoreGuard is a lightweight and propagative authorization module residing in TEE. Extensive experiments show that CoreGuard achieves the same security protection as the black-box security guarantees with negligible overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07109v2">I Want to Break Free! Persuasion and Anti-Social Behavior of LLMs in Multi-Agent Settings with Social Hierarchy</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM)-based agents become increasingly autonomous and will more freely interact with each other, studying interactions between them becomes crucial to anticipate emergent phenomena and potential risks. Drawing inspiration from the widely popular Stanford Prison Experiment, we contribute to this line of research by studying interaction patterns of LLM agents in a context characterized by strict social hierarchy. We do so by specifically studying two types of phenomena: persuasion and anti-social behavior in simulated scenarios involving a guard and a prisoner agent who seeks to achieve a specific goal (i.e., obtaining additional yard time or escape from prison). Leveraging 200 experimental scenarios for a total of 2,000 machine-machine conversations across five different popular LLMs, we provide a set of noteworthy findings. We first document how some models consistently fail in carrying out a conversation in our multi-agent setup where power dynamics are at play. Then, for the models that were able to engage in successful interactions, we empirically show how the goal that an agent is set to achieve impacts primarily its persuasiveness, while having a negligible effect with respect to the agent's anti-social behavior. Third, we highlight how agents' personas, and particularly the guard's personality, drive both the likelihood of successful persuasion from the prisoner and the emergence of anti-social behaviors. Fourth, we show that even without explicitly prompting for specific personalities, anti-social behavior emerges by simply assigning agents' roles. These results bear implications for the development of interactive LLM agents as well as the debate on their societal impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12329v1">Understanding the Role of LLMs in Multimodal Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The rapid advancement of Multimodal Large Language Models (MLLMs) has been accompanied by the development of various benchmarks to evaluate their capabilities. However, the true nature of these evaluations and the extent to which they assess multimodal reasoning versus merely leveraging the underlying Large Language Model (LLM) backbone remain unclear. This paper presents a comprehensive investigation into the role of LLM backbones in MLLM evaluation, focusing on two critical aspects: the degree to which current benchmarks truly assess multimodal reasoning and the influence of LLM prior knowledge on performance. Specifically, we introduce a modified evaluation protocol to disentangle the contributions of the LLM backbone from multimodal integration, and an automatic knowledge identification technique for diagnosing whether LLMs equip the necessary knowledge for corresponding multimodal questions. Our study encompasses four diverse MLLM benchmarks and eight state-of-the-art MLLMs. Key findings reveal that some benchmarks allow high performance even without visual inputs and up to 50\% of error rates can be attributed to insufficient world knowledge in the LLM backbone, indicating a heavy reliance on language capabilities. To address knowledge deficiencies, we propose a knowledge augmentation pipeline that achieves significant performance gains, with improvements of up to 60\% on certain datasets, resulting in a approximately 4x increase in performance. Our work provides crucial insights into the role of the LLM backbone in MLLMs, and highlights the need for more nuanced benchmarking approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12318v1">UTF:Undertrained Tokens as Fingerprints A Novel Approach to LLM Identification</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Fingerprinting large language models (LLMs) is essential for verifying model ownership, ensuring authenticity, and preventing misuse. Traditional fingerprinting methods often require significant computational overhead or white-box verification access. In this paper, we introduce UTF, a novel and efficient approach to fingerprinting LLMs by leveraging under-trained tokens. Under-trained tokens are tokens that the model has not fully learned during its training phase. By utilizing these tokens, we perform supervised fine-tuning to embed specific input-output pairs into the model. This process allows the LLM to produce predetermined outputs when presented with certain inputs, effectively embedding a unique fingerprint. Our method has minimal overhead and impact on model's performance, and does not require white-box access to target model's ownership identification. Compared to existing fingerprinting methods, UTF is also more effective and robust to fine-tuning and random guess.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13993v2">Exploring Changes in Nation Perception with Nationality-Assigned Personas in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Pre-print, Under review
    </div>
    <details class="paper-abstract">
      Persona assignment has become a common strategy for customizing LLM use to particular tasks and contexts. In this study, we explore how evaluation of different nations change when LLMs are assigned specific nationality personas. We assign 193 different nationality personas (e.g., an American person) to four LLMs and examine how the LLM evaluations (or ''perceptions'')of countries change. We find that all LLM-persona combinations tend to favor Western European nations, though nation-personas push LLM behaviors to focus more on and treat the nation-persona's own region more favorably. Eastern European, Latin American, and African nations are treated more negatively by different nationality personas. We additionally find that evaluations by nation-persona LLMs of other nations correlate with human survey responses but fail to match the values closely. Our study provides insight into how biases and stereotypes are realized within LLMs when adopting different national personas. In line with the ''Blueprint for an AI Bill of Rights'', our findings underscore the critical need for developing mechanisms to ensure that LLM outputs promote fairness and avoid over-generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05581v2">Adaptation Odyssey in LLMs: Why Does Additional Pretraining Sometimes Fail to Improve?</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Accepted to EMNLP 2024 Main Conference
    </div>
    <details class="paper-abstract">
      In the last decade, the generalization and adaptation abilities of deep learning models were typically evaluated on fixed training and test distributions. Contrary to traditional deep learning, large language models (LLMs) are (i) even more overparameterized, (ii) trained on unlabeled text corpora curated from the Internet with minimal human intervention, and (iii) trained in an online fashion. These stark contrasts prevent researchers from transferring lessons learned on model generalization and adaptation in deep learning contexts to LLMs. To this end, our short paper introduces empirical observations that aim to shed light on further training of already pretrained language models. Specifically, we demonstrate that training a model on a text domain could degrade its perplexity on the test portion of the same domain. We observe with our subsequent analysis that the performance degradation is positively correlated with the similarity between the additional and the original pretraining dataset of the LLM. Our further token-level perplexity observations reveals that the perplexity degradation is due to a handful of tokens that are not informative about the domain. We hope these findings will guide us in determining when to adapt a model vs when to rely on its foundational capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12299v1">Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance across many tasks, yet aligning them with desired behaviors remains challenging. Activation intervention has emerged as an effective and economical method to modify the behavior of LLMs. Despite considerable interest in this area, current intervention methods exclusively employ a fixed steering vector to modify model activations, lacking adaptability to diverse input semantics. To address this limitation, we propose Semantics-Adaptive Dynamic Intervention (SADI), a novel method that constructs a dynamic steering vector to intervene model activations at inference time. More specifically, SADI utilizes activation differences in contrastive pairs to precisely identify critical elements of an LLM (i.e., attention heads, hidden states, and neurons) for targeted intervention. During inference, SADI dynamically steers model behavior by scaling element-wise activations based on the directions of input semantics. Experimental results show that SADI outperforms established baselines by substantial margins, improving task performance without training. SADI's cost-effectiveness and generalizability across various LLM backbones and tasks highlight its potential as a versatile alignment technique. In addition, we release the code to foster research along this line:https://github.com/weixuan-wang123/SADI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12508v2">MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation for Text-Video Retrieval-Rerank Pipeline</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 EMNLP 2024 Industry Track Accepted (Camera-Ready Version)
    </div>
    <details class="paper-abstract">
      The rapid expansion of multimedia content has made accurately retrieving relevant videos from large collections increasingly challenging. Recent advancements in text-video retrieval have focused on cross-modal interactions, large-scale foundation model training, and probabilistic modeling, yet often neglect the crucial user perspective, leading to discrepancies between user queries and the content retrieved. To address this, we introduce MERLIN (Multimodal Embedding Refinement via LLM-based Iterative Navigation), a novel, training-free pipeline that leverages Large Language Models (LLMs) for iterative feedback learning. MERLIN refines query embeddings from a user perspective, enhancing alignment between queries and video content through a dynamic question answering process. Experimental results on datasets like MSR-VTT, MSVD, and ActivityNet demonstrate that MERLIN substantially improves Recall@1, outperforming existing systems and confirming the benefits of integrating LLMs into multimodal retrieval systems for more responsive and context-aware multimedia retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.00024v5">Can LLMs Patch Security Issues?</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive proficiency in code generation. Unfortunately, these models share a weakness with their human counterparts: producing code that inadvertently has security vulnerabilities. These vulnerabilities could allow unauthorized attackers to access sensitive data or systems, which is unacceptable for safety-critical applications. In this work, we propose Feedback-Driven Security Patching (FDSP), where LLMs automatically refine generated, vulnerable code. Our approach leverages automatic static code analysis to empower the LLM to generate and implement potential solutions to address vulnerabilities. We address the research communitys needs for safe code generation by introducing a large-scale dataset, PythonSecurityEval, covering the diversity of real-world applications, including databases, websites and operating systems. We empirically validate that FDSP outperforms prior work that uses self-feedback from LLMs by up to 17.6% through our procedure that injects targeted, external feedback. Code and data are available at \url{https://github.com/Kamel773/LLM-code-refine}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00333v2">A Practice-Friendly LLM-Enhanced Paradigm with Preference Parsing for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      The training paradigm integrating large language models (LLM) is gradually reshaping sequential recommender systems (SRS) and has shown promising results. However, most existing LLM-enhanced methods rely on rich textual information on the item side and instance-level supervised fine-tuning (SFT) to inject collaborative information into LLM, which is inefficient and limited in many applications. To alleviate these problems, this paper proposes a practice-friendly LLM-enhanced paradigm with preference parsing (P2Rec) for SRS. Specifically, in the information reconstruction stage, we design a new user-level SFT task for collaborative information injection with the assistance of a pre-trained SRS model, which is more efficient and compatible with limited text information. Our goal is to let LLM learn to reconstruct a corresponding prior preference distribution from each user's interaction sequence, where LLM needs to effectively parse the latent category of each item and the relationship between different items to accomplish this task. In the information augmentation stage, we feed each item into LLM to obtain a set of enhanced embeddings that combine collaborative information and LLM inference capabilities. These embeddings can then be used to help train various future SRS models. Finally, we verify the effectiveness and efficiency of our TSLRec on three SRS benchmark datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06949v2">Seeker: Enhancing Exception Handling in Code with LLM-based Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 26 pages, 7 figures. Submitted ICLR 2025
    </div>
    <details class="paper-abstract">
      In real world software development, improper or missing exception handling can severely impact the robustness and reliability of code. Exception handling mechanisms require developers to detect, capture, and manage exceptions according to high standards, but many developers struggle with these tasks, leading to fragile code. This problem is particularly evident in open source projects and impacts the overall quality of the software ecosystem. To address this challenge, we explore the use of large language models (LLMs) to improve exception handling in code. Through extensive analysis, we identify three key issues: Insensitive Detection of Fragile Code, Inaccurate Capture of Exception Types, and Distorted Handling Solutions. These problems are widespread across real world repositories, suggesting that robust exception handling practices are often overlooked or mishandled. In response, we propose Seeker, a multi agent framework inspired by expert developer strategies for exception handling. Seeker uses agents: Scanner, Detector, Predator, Ranker, and Handler to assist LLMs in detecting, capturing, and resolving exceptions more effectively. Our work is the first systematic study on leveraging LLMs to enhance exception handling practices, providing valuable insights for future improvements in code reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17232v2">Beyond Demographics: Aligning Role-playing LLM-based Agents Using Human Belief Networks</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Creating human-like large language model (LLM) agents is crucial for faithful social simulation. Having LLMs role-play based on demographic information sometimes improves human likeness but often does not. This study assessed whether LLM alignment with human behavior can be improved by integrating information from empirically-derived human belief networks. Using data from a human survey, we estimated a belief network encompassing 64 topics loading on nine non-overlapping latent factors. We then seeded LLM-based agents with an opinion on one topic, and assessed the alignment of its expressed opinions on remaining test topics with corresponding human data. Role-playing based on demographic information alone did not align LLM and human opinions, but seeding the agent with a single belief greatly improved alignment for topics related in the belief network, and not for topics outside the network. These results suggest a novel path for human-LLM belief alignment in work seeking to simulate and understand patterns of belief distributions in society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12207v1">Divide-Verify-Refine: Aligning LLM Responses with Complex Instructions</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Recent studies show that LLMs, particularly open-source models, struggle to follow complex instructions with multiple constraints. Despite the importance, methods to improve LLMs' adherence to such constraints remain unexplored, and current research focuses on evaluating this ability rather than developing solutions. While a few studies enhance constraint adherence through model tuning, this approach is computationally expensive and heavily reliant on training data quality. An alternative is to leverage LLMs' self-correction capabilities, allowing them to adjust responses to better meet specified constraints. However, this self-correction ability of LLMs is limited by the feedback quality, as LLMs cannot autonomously generate reliable feedback or detect errors. Moreover, the self-refinement process heavily depends on few-shot examples that illustrate how to modify responses to meet constraints. As constraints in complex instructions are diverse and vary widely, manually crafting few-shot examples for each constraint type can be labor-intensive and sub-optimal. To deal with these two challenges, we propose the Divide-Verify-Refine (DVR) framework with three steps: (1) Divide complex instructions into single constraints and prepare appropriate tools; (2) Verify: To address the feedback quality problem, these tools will rigorously verify responses and provide reliable feedback; (3) Refine: To address the constraint diversity challenge, we design a refinement repository that collects successful refinement processes and uses them as few-shot demonstrations for future cases, allowing LLMs to learn from the past experience during inference. Additionally, we develop a new dataset of complex instructions, each containing 1-6 constraints. Experiments show that the framework significantly improves performance, doubling LLama3.1-8B's constraint adherence on instructions with 6 constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07427v2">Beyond Inter-Item Relations: Dynamic Adaption for Enhancing LLM-Based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 11 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Sequential recommender systems (SRS) predict the next items that users may prefer based on user historical interaction sequences. Inspired by the rise of large language models (LLMs) in various AI applications, there is a surge of work on LLM-based SRS. Despite their attractive performance, existing LLM-based SRS still exhibit some limitations, including neglecting intra-item relations, ignoring long-term collaborative knowledge and using inflexible architecture designs for adaption. To alleviate these issues, we propose an LLM-based sequential recommendation model named DARec. Built on top of coarse-grained adaption for capturing inter-item relations, DARec is further enhanced with (1) context masking that models intra-item relations to help LLM better understand token and item semantics in the context of SRS, (2) collaborative knowledge injection that helps LLM incorporate long-term collaborative knowledge, and (3) a dynamic adaption mechanism that uses Bayesian optimization to flexibly choose layer-wise adapter architectures in order to better incorporate different sequential information. Extensive experiments demonstrate that DARec can effectively handle sequential recommendation in a dynamic and adaptive manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12168v1">COMET: Towards Partical W4A4KV4 LLMs Serving</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 14 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Quantization is a widely-used compression technology to reduce the overhead of serving large language models (LLMs) on terminal devices and in cloud data centers. However, prevalent quantization methods, such as 8-bit weight-activation or 4-bit weight-only quantization, achieve limited performance improvements due to poor support for low-precision (e.g., 4-bit) activation. This work, for the first time, realizes practical W4A4KV4 serving for LLMs, fully utilizing the INT4 tensor cores on modern GPUs and reducing the memory bottleneck caused by the KV cache. Specifically, we propose a novel fine-grained mixed-precision quantization algorithm (FMPQ) that compresses most activations into 4-bit with negligible accuracy loss. To support mixed-precision matrix multiplication for W4A4 and W4A8, we develop a highly optimized W4Ax kernel. Our approach introduces a novel mixed-precision data layout to facilitate access and fast dequantization for activation and weight tensors, utilizing the GPU's software pipeline to hide the overhead of data loading and conversion. Additionally, we propose fine-grained streaming multiprocessor (SM) scheduling to achieve load balance across different SMs. We integrate the optimized W4Ax kernel into our inference framework, COMET, and provide efficient management to support popular LLMs such as LLaMA-3-70B. Extensive evaluations demonstrate that, when running LLaMA family models on a single A100-80G-SMX4, COMET achieves a kernel-level speedup of \textbf{$2.88\times$} over cuBLAS and a \textbf{$2.02 \times$} throughput improvement compared to TensorRT-LLM from an end-to-end framework perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10814v2">Your Mixture-of-Experts LLM Is Secretly an Embedding Model For Free</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Code: https://github.com/tianyi-lab/MoE-Embedding
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) excel on generation tasks, their decoder-only architecture often limits their potential as embedding models if no further representation finetuning is applied. Does this contradict their claim of generalists? To answer the question, we take a closer look at Mixture-of-Experts (MoE) LLMs. Our study shows that the expert routers in MoE LLMs can serve as an off-the-shelf embedding model with promising performance on a diverse class of embedding-focused tasks, without requiring any finetuning. Moreover, our extensive analysis shows that the MoE routing weights (RW) is complementary to the hidden state (HS) of LLMs, a widely-used embedding. Compared to HS, we find that RW is more robust to the choice of prompts and focuses on high-level semantics. Motivated by the analysis, we propose MoEE combining RW and HS, which achieves better performance than using either separately. Our exploration of their combination and prompting strategy shed several novel insights, e.g., a weighted sum of RW and HS similarities outperforms the similarity on their concatenation. Our experiments are conducted on 6 embedding tasks with 20 datasets from the Massive Text Embedding Benchmark (MTEB). The results demonstrate the significant improvement brought by MoEE to LLM-based embedding without further finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11693v2">IntGrad MT: Eliciting LLMs' Machine Translation Capabilities with Sentence Interpolation and Gradual MT</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Recent Large Language Models (LLMs) have demonstrated strong performance in translation without needing to be finetuned on additional parallel corpora. However, they still underperform for low-resource language pairs. Previous works have focused on mitigating this issue by leveraging relevant few-shot examples or external resources such as dictionaries or grammar books, making models heavily reliant on these nonparametric sources of information. In this paper, we propose a novel method named IntGrad MT that focuses on fully exploiting an LLM's inherent translation capability. IntGrad MT achieves this by constructing a chain of few-shot examples, each consisting of a source sentence and the model's own translation, that rise incrementally in difficulty. IntGrad MT employs two techniques: Sentence Interpolation, which generates a sequence of sentences that gradually change from an easy sentence to translate to a difficult one, and Gradual MT, which sequentially translates this chain using translations of earlier sentences as few-shot examples for the translation of subsequent ones. With this approach, we observe a substantial enhancement in the xCOMET scores of various LLMs for multiple languages, especially in low-resource languages such as Hindi(8.26), Swahili(7.10), Bengali(6.97) and Marathi(13.03). Our approach presents a practical way of enhancing LLMs' performance without extra training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12154v1">Exploiting LLMs' Reasoning Capability to Infer Implicit Concepts in Legal Information Retrieval</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)
    </div>
    <details class="paper-abstract">
      Statutory law retrieval is a typical problem in legal language processing, that has various practical applications in law engineering. Modern deep learning-based retrieval methods have achieved significant results for this problem. However, retrieval systems relying on semantic and lexical correlations often exhibit limitations, particularly when handling queries that involve real-life scenarios, or use the vocabulary that is not specific to the legal domain. In this work, we focus on overcoming this weaknesses by utilizing the logical reasoning capabilities of large language models (LLMs) to identify relevant legal terms and facts related to the situation mentioned in the query. The proposed retrieval system integrates additional information from the term--based expansion and query reformulation to improve the retrieval accuracy. The experiments on COLIEE 2022 and COLIEE 2023 datasets show that extra knowledge from LLMs helps to improve the retrieval result of both lexical and semantic ranking models. The final ensemble retrieval system outperformed the highest results among all participating teams in the COLIEE 2022 and 2023 competitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15513v2">PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 a sibling project to SafeRLHF and BeaverTails
    </div>
    <details class="paper-abstract">
      In this work, we introduce the PKU-SafeRLHF dataset, designed to promote research on safety alignment in large language models (LLMs). As a sibling project to SafeRLHF and BeaverTails, we separate annotations of helpfulness and harmlessness for question-answering pairs, providing distinct perspectives on these coupled attributes. Overall, we provide 44.6k refined prompts and 265k question-answer pairs with safety meta-labels for 19 harm categories and three severity levels ranging from minor to severe, with answers generated by Llama-family models. Based on this, we collected 166.8k preference data, including dual-preference (helpfulness and harmlessness decoupled) and single-preference data (trade-off the helpfulness and harmlessness from scratch), respectively. Using the large-scale annotation data, we further train severity-sensitive moderation for the risk control of LLMs and safety-centric RLHF algorithms for the safety alignment of LLMs. We believe this dataset will be a valuable resource for the community, aiding in the safe deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12153v1">Layer-of-Thoughts Prompting (LoT): Leveraging LLM-Based Retrieval with Constraint Hierarchies</a></div>
    <div class="paper-meta">
      📅 2024-10-16
      | 💬 Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach termed Layer-of-Thoughts Prompting (LoT), which utilizes constraint hierarchies to filter and refine candidate responses to a given query. By integrating these constraints, our method enables a structured retrieval process that enhances explainability and automation. Existing methods have explored various prompting techniques but often present overly generalized frameworks without delving into the nuances of prompts in multi-turn interactions. Our work addresses this gap by focusing on the hierarchical relationships among prompts. We demonstrate that the efficacy of thought hierarchy plays a critical role in developing efficient and interpretable retrieval algorithms. Leveraging Large Language Models (LLMs), LoT significantly improves the accuracy and comprehensibility of information retrieval tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10714v2">SeedLM: Compressing LLM Weights into Seeds of Pseudo-Random Generators</a></div>
    <div class="paper-meta">
      📅 2024-10-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed natural language processing, but face significant challenges in widespread deployment due to their high runtime cost. In this paper, we introduce SeedLM, a novel post-training compression method that uses seeds of pseudo-random generators to encode and compress model weights. Specifically, for each block of weights, we find a seed that is fed into a Linear Feedback Shift Register (LFSR) during inference to efficiently generate a random matrix. This matrix is then linearly combined with compressed coefficients to reconstruct the weight block. SeedLM reduces memory access and leverages idle compute cycles during inference, effectively speeding up memory-bound tasks by trading compute for fewer memory accesses. Unlike state-of-the-art compression methods that rely on calibration data, our approach is data-free and generalizes well across diverse tasks. Our experiments with Llama 3 70B, which is particularly challenging to compress, show that SeedLM achieves significantly better zero-shot accuracy retention at 4- and 3-bit than state-of-the-art techniques, while maintaining performance comparable to FP16 baselines. Additionally, FPGA-based tests demonstrate that 4-bit SeedLM, as model size increases to 70B, approaches a 4x speed-up over an FP16 Llama 2/3 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09572v4">Eyes Closed, Safety On: Protecting Multimodal LLMs via Image-to-Text Transformation</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 ECCV2024 (Project Page: https://gyhdog99.github.io/projects/ecso/)
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) have shown impressive reasoning abilities. However, they are also more vulnerable to jailbreak attacks than their LLM predecessors. Although still capable of detecting the unsafe responses, we observe that safety mechanisms of the pre-aligned LLMs in MLLMs can be easily bypassed with the introduction of image features. To construct robust MLLMs, we propose ECSO (Eyes Closed, Safety On), a novel training-free protecting approach that exploits the inherent safety awareness of MLLMs, and generates safer responses via adaptively transforming unsafe images into texts to activate the intrinsic safety mechanism of pre-aligned LLMs in MLLMs. Experiments on five state-of-the-art (SoTA) MLLMs demonstrate that ECSO enhances model safety significantly (e.g.,, 37.6% improvement on the MM-SafetyBench (SD+OCR) and 71.3% on VLSafe with LLaVA-1.5-7B), while consistently maintaining utility results on common MLLM benchmarks. Furthermore, we show that ECSO can be used as a data engine to generate supervised-finetuning (SFT) data for MLLM alignment without extra human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11239v1">HR-Agent: A Task-Oriented Dialogue (TOD) LLM Agent Tailored for HR Applications</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Recent LLM (Large Language Models) advancements benefit many fields such as education and finance, but HR has hundreds of repetitive processes, such as access requests, medical claim filing and time-off submissions, which are unaddressed. We relate these tasks to the LLM agent, which has addressed tasks such as writing assisting and customer support. We present HR-Agent, an efficient, confidential, and HR-specific LLM-based task-oriented dialogue system tailored for automating repetitive HR processes such as medical claims and access requests. Since conversation data is not sent to an LLM during inference, it preserves confidentiality required in HR-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11235v1">Unleashing the Power of LLMs as Multi-Modal Encoders for Text and Graph-Structured Data</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Graph-structured information offers rich contextual information that can enhance language models by providing structured relationships and hierarchies, leading to more expressive embeddings for various applications such as retrieval, question answering, and classification. However, existing methods for integrating graph and text embeddings, often based on Multi-layer Perceptrons (MLPs) or shallow transformers, are limited in their ability to fully exploit the heterogeneous nature of these modalities. To overcome this, we propose Janus, a simple yet effective framework that leverages Large Language Models (LLMs) to jointly encode text and graph data. Specifically, Janus employs an MLP adapter to project graph embeddings into the same space as text embeddings, allowing the LLM to process both modalities jointly. Unlike prior work, we also introduce contrastive learning to align the graph and text spaces more effectively, thereby improving the quality of learned joint embeddings. Empirical results across six datasets spanning three tasks, knowledge graph-contextualized question answering, graph-text pair classification, and retrieval, demonstrate that Janus consistently outperforms existing baselines, achieving significant improvements across multiple datasets, with gains of up to 11.4% in QA tasks. These results highlight Janus's effectiveness in integrating graph and text data. Ablation studies further validate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10456v2">Ada-K Routing: Boosting the Efficiency of MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Coauthors do not reach a consensus on submitting the current version
    </div>
    <details class="paper-abstract">
      In the era of Large Language Models (LLMs), Mixture-of-Experts (MoE) architectures offer a promising approach to managing computational costs while scaling up model parameters. Conventional MoE-based LLMs typically employ static Top-K routing, which activates a fixed and equal number of experts for each token regardless of their significance within the context. In this paper, we propose a novel Ada-K routing strategy that dynamically adjusts the number of activated experts for each token, thereby improving the balance between computational efficiency and model performance. Specifically, our strategy incorporates learnable and lightweight allocator modules that decide customized expert resource allocation tailored to the contextual needs for each token. These allocators are designed to be fully pluggable, making it broadly applicable across all mainstream MoE-based LLMs. We leverage the Proximal Policy Optimization (PPO) algorithm to facilitate an end-to-end learning process for this non-differentiable decision-making framework. Extensive evaluations on four popular baseline models demonstrate that our Ada-K routing method significantly outperforms conventional Top-K routing. Compared to Top-K, our method achieves over 25% reduction in FLOPs and more than 20% inference speedup while still improving performance across various benchmarks. Moreover, the training of Ada-K is highly efficient. Even for Mixtral-8x22B, a MoE-based LLM with more than 140B parameters, the training time is limited to 8 hours. Detailed analysis shows that harder tasks, middle layers, and content words tend to activate more experts, providing valuable insights for future adaptive MoE system designs. Both the training code and model checkpoints will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11906v1">Empowering Users in Digital Privacy Management through Interactive LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      This paper presents a novel application of large language models (LLMs) to enhance user comprehension of privacy policies through an interactive dialogue agent. We demonstrate that LLMs significantly outperform traditional models in tasks like Data Practice Identification, Choice Identification, Policy Summarization, and Privacy Question Answering, setting new benchmarks in privacy policy analysis. Building on these findings, we introduce an innovative LLM-based agent that functions as an expert system for processing website privacy policies, guiding users through complex legal language without requiring them to pose specific questions. A user study with 100 participants showed that users assisted by the agent had higher comprehension levels (mean score of 2.6 out of 3 vs. 1.8 in the control group), reduced cognitive load (task difficulty ratings of 3.2 out of 10 vs. 7.8), increased confidence in managing privacy, and completed tasks in less time (5.5 minutes vs. 15.8 minutes). This work highlights the potential of LLM-based agents to transform user interaction with privacy policies, leading to more informed consent and empowering users in the digital services landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.11511v3">ComplexityNet: Increasing LLM Inference Efficiency by Learning Task Complexity</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      We present ComplexityNet, a streamlined language model designed for assessing task complexity. This model predicts the likelihood of accurate output by various language models, each with different capabilities. Our initial application of ComplexityNet involves the Mostly Basic Python Problems (MBPP) dataset. We pioneered the creation of the first set of labels to define task complexity. ComplexityNet achieved a notable 79% accuracy in determining task complexity, a significant improvement over the 34% accuracy of the original, non fine-tuned model. Furthermore, ComplexityNet effectively reduces computational resource usage by 90% compared to using the highest complexity model, while maintaining a high code generation accuracy of 86.7%. This study demonstrates that fine-tuning smaller models to categorize tasks based on their complexity can lead to a more balanced trade-off between accuracy and efficiency in the use of Large Language Models. Our findings suggest a promising direction for optimizing LLM applications, especially in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11163v1">Model Swarms: Collaborative Search to Adapt LLM Experts via Swarm Intelligence</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      We propose Model Swarms, a collaborative search algorithm to adapt LLMs via swarm intelligence, the collective behavior guiding individual systems. Specifically, Model Swarms starts with a pool of LLM experts and a utility function. Guided by the best-found checkpoints across models, diverse LLM experts collaboratively move in the weight space and optimize a utility function representing model adaptation objectives. Compared to existing model composition approaches, Model Swarms offers tuning-free model adaptation, works in low-data regimes with as few as 200 examples, and does not require assumptions about specific experts in the swarm or how they should be composed. Extensive experiments demonstrate that Model Swarms could flexibly adapt LLM experts to a single task, multi-task domains, reward models, as well as diverse human interests, improving over 12 model composition baselines by up to 21.0% across tasks and contexts. Further analysis reveals that LLM experts discover previously unseen capabilities in initial checkpoints and that Model Swarms enable the weak-to-strong transition of experts through the collaborative search process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12104v1">To Err is AI : A Case Study Informing LLM Flaw Reporting Practices</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      In August of 2024, 495 hackers generated evaluations in an open-ended bug bounty targeting the Open Language Model (OLMo) from The Allen Institute for AI. A vendor panel staffed by representatives of OLMo's safety program adjudicated changes to OLMo's documentation and awarded cash bounties to participants who successfully demonstrated a need for public disclosure clarifying the intent, capacities, and hazards of model deployment. This paper presents a collection of lessons learned, illustrative of flaw reporting best practices intended to reduce the likelihood of incidents and produce safer large language models (LLMs). These include best practices for safety reporting processes, their artifacts, and safety program staffing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09757v2">Evaluating LLM-driven User-Intent Formalization for Verification-Aware Languages</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Proceedings of the 24th Conference on Formal Methods in Computer Aided Design (FMCAD 2024)
    </div>
    <details class="paper-abstract">
      Verification-aware programming languages such as Dafny and F* provide means to formally specify and prove properties of a program. Although the problem of checking an implementation against a specification can be defined mechanically, there is no algorithmic way of ensuring the correctness of the {\it user-intent formalization for programs}, expressed as a formal specification. This is because intent or requirement is expressed {\it informally} in natural language and the specification is a formal artefact. Despite, the advent of large language models (LLMs) has made tremendous strides bridging the gap between informal intent and formal program implementations recently, driven in large parts by benchmarks and automated metrics for evaluation. Recent work has proposed a framework for evaluating the {\it user-intent formalization} problem for mainstream programming languages~\cite{endres-fse24}. However, such an approach does not readily extend to verification-aware languages that support rich specifications (using quantifiers and ghost variables) that cannot be evaluated through dynamic execution. Previous work also required generating program mutants using LLMs to create the benchmark. We advocate an alternate, perhaps simpler approach of {\it symbolically testing specifications} to provide an intuitive metric for evaluating the quality of specifications for verification-aware languages. We demonstrate that our automated metric agrees closely on a human-labeled dataset of Dafny specifications for the popular MBPP code-generation benchmark, yet demonstrates cases where the human labeling is not perfect. We also outline formal verification challenges that need to be addressed to apply the technique more widely. We believe our work provides a stepping stone to enable the establishment of a benchmark and research agenda for the problem of user-intent formalization for programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12052v1">Skill-LLM: Repurposing General-Purpose LLMs for Skill Extraction</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Accurate skill extraction from job descriptions is crucial in the hiring process but remains challenging. Named Entity Recognition (NER) is a common approach used to address this issue. With the demonstrated success of large language models (LLMs) in various NLP tasks, including NER, we propose fine-tuning a specialized Skill-LLM and a light weight model to improve the precision and quality of skill extraction. In our study, we evaluated the fine-tuned Skill-LLM and the light weight model using a benchmark dataset and compared its performance against state-of-the-art (SOTA) methods. Our results show that this approach outperforms existing SOTA techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19792v1">Substance Beats Style: Why Beginning Students Fail to Code with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      Although LLMs are increasing the productivity of professional programmers, existing work shows that beginners struggle to prompt LLMs to solve text-to-code tasks. Why is this the case? This paper explores two competing hypotheses about the cause of student-LLM miscommunication: (1) students simply lack the technical vocabulary needed to write good prompts, and (2) students do not understand the extent of information that LLMs need to solve code generation tasks. We study (1) with a causal intervention experiment on technical vocabulary and (2) by analyzing graphs that abstract how students edit prompts and the different failures that they encounter. We find that substance beats style: a poor grasp of technical vocabulary is merely correlated with prompt failure; that the information content of prompts predicts success; that students get stuck making trivial edits; and more. Our findings have implications for the use of LLMs in programming education, and for efforts to make computing more accessible with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12048v1">Boosting Logical Fallacy Reasoning in LLMs via Logical Structure Tree</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Logical fallacy uses invalid or faulty reasoning in the construction of a statement. Despite the prevalence and harmfulness of logical fallacies, detecting and classifying logical fallacies still remains a challenging task. We observe that logical fallacies often use connective words to indicate an intended logical relation between two arguments, while the argument semantics does not actually support the logical relation. Inspired by this observation, we propose to build a logical structure tree to explicitly represent and track the hierarchical logic flow among relation connectives and their arguments in a statement. Specifically, this logical structure tree is constructed in an unsupervised manner guided by the constituency tree and a taxonomy of connectives for ten common logical relations, with relation connectives as non-terminal nodes and textual arguments as terminal nodes, and the latter are mostly elementary discourse units. We further develop two strategies to incorporate the logical structure tree into LLMs for fallacy reasoning. Firstly, we transform the tree into natural language descriptions and feed the textualized tree into LLMs as a part of the hard text prompt. Secondly, we derive a relation-aware tree embedding and insert the tree embedding into LLMs as a soft prompt. Experiments on benchmark datasets demonstrate that our approach based on logical structure tree significantly improves precision and recall for both fallacy detection and fallacy classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15198v3">Do LLM Agents Exhibit Social Behavior?</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      As LLMs increasingly take on roles in human-AI interactions and autonomous AI systems, understanding their social behavior becomes important for informed use and continuous improvement. However, their behaviors in social interactions with humans and other agents, as well as the mechanisms shaping their responses, remain underexplored. To address this gap, we introduce a novel probabilistic framework, State-Understanding-Value-Action (SUVA), to systematically analyze LLM responses in social contexts based on their textual outputs (i.e., utterances). Using canonical behavioral economics games and social preference concepts relatable to LLM users, SUVA assesses LLMs' social behavior through both their final decisions and the response generation processes leading to those decisions. Our analysis of eight LLMs -- including two GPT, four LLaMA, and two Mistral models -- suggests that most models do not generate decisions aligned solely with self-interest; instead, they often produce responses that reflect social welfare considerations and display patterns consistent with direct and indirect reciprocity. Additionally, higher-capacity models more frequently display group identity effects. The SUVA framework also provides explainable tools -- including tree-based visualizations and probabilistic dependency analysis -- to elucidate how factors in LLMs' utterance-based reasoning influence their decisions. We demonstrate that utterance-based reasoning reliably predicts LLMs' final actions; references to altruism, fairness, and cooperation in the reasoning increase the likelihood of prosocial actions, while mentions of self-interest and competition reduce them. Overall, our framework enables practitioners to assess LLMs for applications involving social interactions, and provides researchers with a structured method to interpret how LLM behavior arises from utterance-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12001v1">Impacts of Continued Legal Pre-Training and IFT on LLMs' Latent Representations of Human-Defined Legal Concepts</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      This paper aims to offer AI & Law researchers and practitioners a more detailed understanding of whether and how continued pre-training and instruction fine-tuning (IFT) of large language models (LLMs) on legal corpora increases their utilization of human-defined legal concepts when developing global contextual representations of input sequences. We compared three models: Mistral 7B, SaulLM-7B-Base (Mistral 7B with continued pre-training on legal corpora), and SaulLM-7B-Instruct (with further IFT). This preliminary assessment examined 7 distinct text sequences from recent AI & Law literature, each containing a human-defined legal concept. We first compared the proportions of total attention the models allocated to subsets of tokens representing the legal concepts. We then visualized patterns of raw attention score alterations, evaluating whether legal training introduced novel attention patterns corresponding to structures of human legal knowledge. This inquiry revealed that (1) the impact of legal training was unevenly distributed across the various human-defined legal concepts, and (2) the contextual representations of legal knowledge learned during legal training did not coincide with structures of human-defined legal concepts. We conclude with suggestions for further investigation into the dynamics of legal LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02257v3">MMLU-Pro+: Evaluating Higher-Order Reasoning and Shortcut Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted to NeurIPS 2024, Safe Generative AI
    </div>
    <details class="paper-abstract">
      Existing benchmarks for large language models (LLMs) increasingly struggle to differentiate between top-performing models, underscoring the need for more challenging evaluation frameworks. We introduce MMLU-Pro+, an enhanced benchmark building upon MMLU-Pro to assess shortcut learning and higher-order reasoning in LLMs. By incorporating questions with multiple correct answers across diverse domains, MMLU-Pro+ tests LLMs' ability to engage in complex reasoning and resist simplistic problem-solving strategies. Our results show that MMLU-Pro+ maintains MMLU-Pro's difficulty while providing a more rigorous test of model discrimination, particularly in multi-correct answer scenarios. We introduce novel metrics like shortcut selection ratio and correct pair identification ratio, offering deeper insights into model behavior and anchoring bias. Evaluations of six state-of-the-art LLMs reveal significant performance gaps, highlighting variations in reasoning abilities and bias susceptibility. We release the dataset and evaluation codes at \url{https://github.com/asgsaeid/mmlu-pro-plus}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12881v1">MIND: Math Informed syNthetic Dialogues for Pretraining LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 31 pages, 5 figures, 14 tables
    </div>
    <details class="paper-abstract">
      The utility of synthetic data to enhance pretraining data quality and hence to improve downstream task accuracy has been widely explored in recent large language models (LLMs). Yet, these approaches fall inadequate in complex, multi-hop and mathematical reasoning tasks as the synthetic data typically fails to add complementary knowledge to the existing raw corpus. In this work, we propose a novel large-scale and diverse Math Informed syNthetic Dialogue (MIND) generation method that improves the mathematical reasoning ability of LLMs. Specifically, using MIND, we generate synthetic conversations based on OpenWebMath (OWM), resulting in a new math corpus, MIND-OWM. Our experiments with different conversational settings reveal that incorporating knowledge gaps between dialog participants is essential for generating high-quality math data. We further identify an effective way to format and integrate synthetic and raw data during pretraining to maximize the gain in mathematical reasoning, emphasizing the need to restructure raw data rather than use it as-is. Compared to pretraining just on raw data, a model pretrained on MIND-OWM shows significant boost in mathematical reasoning (GSM8K: +13.42%, MATH: +2.30%), including superior performance in specialized knowledge (MMLU: +4.55%, MMLU-STEM: +4.28%) and general purpose reasoning tasks (GENERAL REASONING: +2.51%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11815v1">SGEdit: Bridging LLM with Text2Image Generative Model for Scene Graph-based Image Editing</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 Accepted by ACM Transactions on Graphics and SIGGRAPH Asia 2024. Project page: https://bestzzhang.github.io/SGEdit
    </div>
    <details class="paper-abstract">
      Scene graphs offer a structured, hierarchical representation of images, with nodes and edges symbolizing objects and the relationships among them. It can serve as a natural interface for image editing, dramatically improving precision and flexibility. Leveraging this benefit, we introduce a new framework that integrates large language model (LLM) with Text2Image generative model for scene graph-based image editing. This integration enables precise modifications at the object level and creative recomposition of scenes without compromising overall image integrity. Our approach involves two primary stages: 1) Utilizing a LLM-driven scene parser, we construct an image's scene graph, capturing key objects and their interrelationships, as well as parsing fine-grained attributes such as object masks and descriptions. These annotations facilitate concept learning with a fine-tuned diffusion model, representing each object with an optimized token and detailed description prompt. 2) During the image editing phase, a LLM editing controller guides the edits towards specific areas. These edits are then implemented by an attention-modulated diffusion editor, utilizing the fine-tuned model to perform object additions, deletions, replacements, and adjustments. Through extensive experiments, we demonstrate that our framework significantly outperforms existing image editing methods in terms of editing precision and scene aesthetics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03730v2">Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-15
    </div>
    <details class="paper-abstract">
      We present two multilingual LLMs designed to embrace Europe's linguistic diversity by supporting all 24 official languages of the European Union. Trained on a dataset comprising around 60% non-English data and utilizing a custom multilingual tokenizer, our models address the limitations of existing LLMs that predominantly focus on English or a few high-resource languages. We detail the models' development principles, i.e., data composition, tokenizer optimization, and training methodologies. The models demonstrate competitive performance across multilingual benchmarks, as evidenced by their performance on European versions of ARC, HellaSwag, MMLU, and TruthfulQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11745v1">Personas with Attitudes: Controlling LLMs for Diverse Data Annotation</a></div>
    <div class="paper-meta">
      📅 2024-10-15
      | 💬 21 pages, 13 figures
    </div>
    <details class="paper-abstract">
      We present a novel approach for enhancing diversity and control in data annotation tasks by personalizing large language models (LLMs). We investigate the impact of injecting diverse persona descriptions into LLM prompts across two studies, exploring whether personas increase annotation diversity and whether the impacts of individual personas on the resulting annotations are consistent and controllable. Our results show that persona-prompted LLMs produce more diverse annotations than LLMs prompted without personas and that these effects are both controllable and repeatable, making our approach a suitable tool for improving data annotation in subjective NLP tasks like toxicity detection.
    </details>
</div>
