# llm - 2024_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11960v1">DB-LLM: Accurate Dual-Binarization for Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the field of natural language processing, while the expensive memory and computation consumption impede their practical deployment. Quantization emerges as one of the most effective methods for improving the computational efficiency of LLMs. However, existing ultra-low-bit quantization always causes severe accuracy drops. In this paper, we empirically relieve the micro and macro characteristics of ultra-low bit quantization and present a novel Dual-Binarization method for LLMs, namely DB-LLM. For the micro-level, we take both the accuracy advantage of 2-bit-width and the efficiency advantage of binarization into account, introducing Flexible Dual Binarization (FDB). By splitting 2-bit quantized weights into two independent sets of binaries, FDB ensures the accuracy of representations and introduces flexibility, utilizing the efficient bitwise operations of binarization while retaining the inherent high sparsity of ultra-low bit quantization. For the macro-level, we find the distortion that exists in the prediction of LLM after quantization, which is specified as the deviations related to the ambiguity of samples. We propose the Deviation-Aware Distillation (DAD) method, enabling the model to focus differently on various samples. Comprehensive experiments show that our DB-LLM not only significantly surpasses the current State-of-The-Art (SoTA) in ultra-low bit quantization (eg, perplexity decreased from 9.64 to 7.23), but also achieves an additional 20\% reduction in computational consumption compared to the SOTA method under the same bit-width. Our code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11886v1">The Colorful Future of LLMs: Evaluating and Improving LLMs as Emotional Supporters for Queer Youth</a></div>
    <div class="paper-meta">
      📅 2024-02-19
    </div>
    <details class="paper-abstract">
      Queer youth face increased mental health risks, such as depression, anxiety, and suicidal ideation. Hindered by negative stigma, they often avoid seeking help and rely on online resources, which may provide incompatible information. Although access to a supportive environment and reliable information is invaluable, many queer youth worldwide have no access to such support. However, this could soon change due to the rapid adoption of Large Language Models (LLMs) such as ChatGPT. This paper aims to comprehensively explore the potential of LLMs to revolutionize emotional support for queers. To this end, we conduct a qualitative and quantitative analysis of LLM's interactions with queer-related content. To evaluate response quality, we develop a novel ten-question scale that is inspired by psychological standards and expert input. We apply this scale to score several LLMs and human comments to posts where queer youth seek advice and share experiences. We find that LLM responses are supportive and inclusive, outscoring humans. However, they tend to be generic, not empathetic enough, and lack personalization, resulting in nonreliable and potentially harmful advice. We discuss these challenges, demonstrate that a dedicated prompt can improve the performance, and propose a blueprint of an LLM-supporter that actively (but sensitively) seeks user context to provide personalized, empathetic, and reliable responses. Our annotated dataset is available for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03630v2">Enhancing LLM-Based Coding Tools through Native Integration of IDE-Derived Static Context</a></div>
    <div class="paper-meta">
      📅 2024-02-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in code completion, as evidenced by their essential roles in developing code assistant services such as Copilot. Being trained on in-file contexts, current LLMs are quite effective in completing code for single source files. However, it is challenging for them to conduct repository-level code completion for large software projects that require cross-file information. Existing research on LLM-based repository-level code completion identifies and integrates cross-file contexts, but it suffers from low accuracy and limited context length of LLMs. In this paper, we argue that Integrated Development Environments (IDEs) can provide direct, accurate and real-time cross-file information for repository-level code completion. We propose IDECoder, a practical framework that leverages IDE native static contexts for cross-context construction and diagnosis results for self-refinement. IDECoder utilizes the rich cross-context information available in IDEs to enhance the capabilities of LLMs of repository-level code completion. We conducted preliminary experiments to validate the performance of IDECoder and observed that this synergy represents a promising trend for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11814v1">An Empirical Evaluation of LLMs for Solving Offensive Security Challenges</a></div>
    <div class="paper-meta">
      📅 2024-02-19
    </div>
    <details class="paper-abstract">
      Capture The Flag (CTF) challenges are puzzles related to computer security scenarios. With the advent of large language models (LLMs), more and more CTF participants are using LLMs to understand and solve the challenges. However, so far no work has evaluated the effectiveness of LLMs in solving CTF challenges with a fully automated workflow. We develop two CTF-solving workflows, human-in-the-loop (HITL) and fully-automated, to examine the LLMs' ability to solve a selected set of CTF challenges, prompted with information about the question. We collect human contestants' results on the same set of questions, and find that LLMs achieve higher success rate than an average human participant. This work provides a comprehensive evaluation of the capability of LLMs in solving real world CTF challenges, from real competition to fully automated workflow. Our results provide references for applying LLMs in cybersecurity education and pave the way for systematic evaluation of offensive cybersecurity capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11801v1">Enhancing Empathetic Response Generation by Augmenting LLMs with Small-scale Empathetic Models</a></div>
    <div class="paper-meta">
      📅 2024-02-19
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Empathetic response generation is increasingly significant in AI, necessitating nuanced emotional and cognitive understanding coupled with articulate response expression. Current large language models (LLMs) excel in response expression; however, they lack the ability to deeply understand emotional and cognitive nuances, particularly in pinpointing fine-grained emotions and their triggers. Conversely, small-scale empathetic models (SEMs) offer strength in fine-grained emotion detection and detailed emotion cause identification. To harness the complementary strengths of both LLMs and SEMs, we introduce a Hybrid Empathetic Framework (HEF). HEF regards SEMs as flexible plugins to improve LLM's nuanced emotional and cognitive understanding. Regarding emotional understanding, HEF implements a two-stage emotion prediction strategy, encouraging LLMs to prioritize primary emotions emphasized by SEMs, followed by other categories, substantially alleviates the difficulties for LLMs in fine-grained emotion detection. Regarding cognitive understanding, HEF employs an emotion cause perception strategy, prompting LLMs to focus on crucial emotion-eliciting words identified by SEMs, thus boosting LLMs' capabilities in identifying emotion causes. This collaborative approach enables LLMs to discern emotions more precisely and formulate empathetic responses. We validate HEF on the Empathetic-Dialogue dataset, and the findings indicate that our framework enhances the refined understanding of LLMs and their ability to convey empathetic responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12298v1">Is Open-Source There Yet? A Comparative Study on Commercial and Open-Source LLMs in Their Ability to Label Chest X-Ray Reports</a></div>
    <div class="paper-meta">
      📅 2024-02-19
    </div>
    <details class="paper-abstract">
      Introduction: With the rapid advances in large language models (LLMs), there have been numerous new open source as well as commercial models. While recent publications have explored GPT-4 in its application to extracting information of interest from radiology reports, there has not been a real-world comparison of GPT-4 to different leading open-source models. Materials and Methods: Two different and independent datasets were used. The first dataset consists of 540 chest x-ray reports that were created at the Massachusetts General Hospital between July 2019 and July 2021. The second dataset consists of 500 chest x-ray reports from the ImaGenome dataset. We then compared the commercial models GPT-3.5 Turbo and GPT-4 from OpenAI to the open-source models Mistral-7B, Mixtral-8x7B, Llama2-13B, Llama2-70B, QWEN1.5-72B and CheXbert and CheXpert-labeler in their ability to accurately label the presence of multiple findings in x-ray text reports using different prompting techniques. Results: On the ImaGenome dataset, the best performing open-source model was Llama2-70B with micro F1-scores of 0.972 and 0.970 for zero- and few-shot prompts, respectively. GPT-4 achieved micro F1-scores of 0.975 and 0.984, respectively. On the institutional dataset, the best performing open-source model was QWEN1.5-72B with micro F1-scores of 0.952 and 0.965 for zero- and few-shot prompting, respectively. GPT-4 achieved micro F1-scores of 0.975 and 0.973, respectively. Conclusion: In this paper, we show that while GPT-4 is superior to open-source models in zero-shot report labeling, the implementation of few-shot prompting can bring open-source models on par with GPT-4. This shows that open-source models could be a performant and privacy preserving alternative to GPT-4 for the task of radiology report classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12264v1">Uncertainty quantification in fine-tuned LLMs using LoRA ensembles</a></div>
    <div class="paper-meta">
      📅 2024-02-19
      | 💬 8 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and model efficacy on the different target domains during and after fine-tuning. In particular, backed by the numerical experiments, we hypothesise about signals from entropic uncertainty measures for data domains that are inherently difficult for a given architecture to learn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12222v1">CovRL: Fuzzing JavaScript Engines with Coverage-Guided Reinforcement Learning for LLM-based Mutation</a></div>
    <div class="paper-meta">
      📅 2024-02-19
      | 💬 14 pages, 4 figures, 9 tables, 2 listings
    </div>
    <details class="paper-abstract">
      Fuzzing is an effective bug-finding technique but it struggles with complex systems like JavaScript engines that demand precise grammatical input. Recently, researchers have adopted language models for context-aware mutation in fuzzing to address this problem. However, existing techniques are limited in utilizing coverage guidance for fuzzing, which is rather performed in a black-box manner. This paper presents a novel technique called CovRL (Coverage-guided Reinforcement Learning) that combines Large Language Models (LLMs) with reinforcement learning from coverage feedback. Our fuzzer, CovRL-Fuzz, integrates coverage feedback directly into the LLM by leveraging the Term Frequency-Inverse Document Frequency (TF-IDF) method to construct a weighted coverage map. This map is key in calculating the fuzzing reward, which is then applied to the LLM-based mutator through reinforcement learning. CovRL-Fuzz, through this approach, enables the generation of test cases that are more likely to discover new coverage areas, thus improving vulnerability detection while minimizing syntax and semantic errors, all without needing extra post-processing. Our evaluation results indicate that CovRL-Fuzz outperforms the state-of-the-art fuzzers in terms of code coverage and bug-finding capabilities: CovRL-Fuzz identified 48 real-world security-related bugs in the latest JavaScript engines, including 39 previously unknown vulnerabilities and 11 CVEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.06452v3">Understanding the Effects of RLHF on LLM Generalisation and Diversity</a></div>
    <div class="paper-meta">
      📅 2024-02-19
      | 💬 Code available here: https://github.com/facebookresearch/rlfh-gen-div
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) fine-tuned with reinforcement learning from human feedback (RLHF) have been used in some of the most widely deployed AI models to date, such as OpenAI's ChatGPT or Anthropic's Claude. While there has been significant work developing these methods, our understanding of the benefits and downsides of each stage in RLHF is still limited. To fill this gap, we present an extensive analysis of how each stage of the process (i.e. supervised fine-tuning (SFT), reward modelling, and RLHF) affects two key properties: out-of-distribution (OOD) generalisation and output diversity. OOD generalisation is crucial given the wide range of real-world scenarios in which these models are being used, while output diversity refers to the model's ability to generate varied outputs and is important for a variety of use cases. We perform our analysis across two base models on both summarisation and instruction following tasks, the latter being highly relevant for current LLM use cases. We find that RLHF generalises better than SFT to new inputs, particularly as the distribution shift between train and test becomes larger. However, RLHF significantly reduces output diversity compared to SFT across a variety of measures, implying a tradeoff in current LLM fine-tuning methods between generalisation and diversity. Our results provide guidance on which fine-tuning method should be used depending on the application, and show that more research is needed to improve the tradeoff between generalisation and diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2401.07324v3">Small LLMs Are Weak Tool Learners: A Multi-LLM Agent</a></div>
    <div class="paper-meta">
      📅 2024-02-19
      | 💬 On progress, github repo: https://github.com/X-PLUG/Multi-LLM-Agent
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete various tasks in a self-directed fashion. The challenge of tool use demands that LLMs not only understand user queries and generate answers accurately but also excel in task planning, tool invocation, and result summarization. While traditional works focus on training a single LLM with all these capabilities, performance limitations become apparent, particularly with smaller models. To overcome these challenges, we propose a novel approach that decomposes the aforementioned capabilities into a planner, caller, and summarizer. Each component is implemented by a single LLM that focuses on a specific capability and collaborates with others to accomplish the task. This modular framework facilitates individual updates and the potential use of smaller LLMs for building each capability. To effectively train this framework, we introduce a two-stage training paradigm. First, we fine-tune a backbone LLM on the entire dataset without discriminating sub-tasks, providing the model with a comprehensive understanding of the task. Second, the fine-tuned LLM is used to instantiate the planner, caller, and summarizer respectively, which are continually fine-tuned on respective sub-tasks. Evaluation across various tool-use benchmarks illustrates that our proposed multi-LLM framework surpasses the traditional single-LLM approach, highlighting its efficacy and advantages in tool learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11712v1">Modelling Political Coalition Negotiations Using LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-02-18
    </div>
    <details class="paper-abstract">
      Coalition negotiations are a cornerstone of parliamentary democracies, characterised by complex interactions and strategic communications among political parties. Despite its significance, the modelling of these negotiations has remained unexplored with the domain of Natural Language Processing (NLP), mostly due to lack of proper data. In this paper, we introduce coalition negotiations as a novel NLP task, and model it as a negotiation between large language model-based agents. We introduce a multilingual dataset, POLCA, comprising manifestos of European political parties and coalition agreements over a number of elections in these countries. This dataset addresses the challenge of the current scope limitations in political negotiation modelling by providing a diverse, real-world basis for simulation. Additionally, we propose a hierarchical Markov decision process designed to simulate the process of coalition negotiation between political parties and predict the outcomes. We evaluate the performance of state-of-the-art large language models (LLMs) as agents in handling coalition negotiations, offering insights into their capabilities and paving the way for future advancements in political modelling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11635v1">Tool-Augmented LLMs as a Universal Interface for IDEs</a></div>
    <div class="paper-meta">
      📅 2024-02-18
      | 💬 First IDE Workshop, ICSE'24
    </div>
    <details class="paper-abstract">
      Modern-day Integrated Development Environments (IDEs) have come a long way from the early text editing utilities to the complex programs encompassing thousands of functions to help developers. However, with the increasing number of efficiency-enhancing tools incorporated, IDEs gradually became sophisticated software with a steep learning curve. The rise of the Large Language Models (LLMs) capable of both natural language dialogue and code generation leads to a discourse on the obsolescence of the concept of IDE. In this work, we offer a view on the place of the LLMs in the IDEs as the universal interface wrapping the IDE facilities. We envision a model that is able to perform complex actions involving multiple IDE features upon user command, stripping the user experience of the tedious work involved in searching through options and actions. For the practical part of the work, we engage with the works exploring the ability of LLMs to call for external tools to expedite a given task execution. We showcase a proof-of-concept of such a tool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11633v1">Self-seeding and Multi-intent Self-instructing LLMs for Generating Intent-aware Information-Seeking dialogs</a></div>
    <div class="paper-meta">
      📅 2024-02-18
    </div>
    <details class="paper-abstract">
      Identifying user intents in information-seeking dialogs is crucial for a system to meet user's information needs. Intent prediction (IP) is challenging and demands sufficient dialogs with human-labeled intents for training. However, manually annotating intents is resource-intensive. While large language models (LLMs) have been shown to be effective in generating synthetic data, there is no study on using LLMs to generate intent-aware information-seeking dialogs. In this paper, we focus on leveraging LLMs for zero-shot generation of large-scale, open-domain, and intent-aware information-seeking dialogs. We propose SOLID, which has novel self-seeding and multi-intent self-instructing schemes. The former improves the generation quality by using the LLM's own knowledge scope to initiate dialog generation; the latter prompts the LLM to generate utterances sequentially, and mitigates the need for manual prompt design by asking the LLM to autonomously adapt its prompt instruction when generating complex multi-intent utterances. Furthermore, we propose SOLID-RL, which is further trained to generate a dialog in one step on the data generated by SOLID. We propose a length-based quality estimation mechanism to assign varying weights to SOLID-generated dialogs based on their quality during the training process of SOLID-RL. We use SOLID and SOLID-RL to generate more than 300k intent-aware dialogs, surpassing the size of existing datasets. Experiments show that IP methods trained on dialogs generated by SOLID and SOLID-RL achieve better IP quality than ones trained on human-generated dialogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11577v1">Extensible Embedding: A Flexible Multipler For LLM's Context Length</a></div>
    <div class="paper-meta">
      📅 2024-02-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) call for extension of context to handle many critical applications. However, the existing approaches are prone to expensive costs and inferior quality of context extension. In this work, we propose Extensible Embedding, which realizes high-quality extension of LLM's context with strong flexibility and cost-effectiveness. Extensible embedding stand as an enhancement of typical token embedding, which represents the information for an extensible scope of context instead of a single token. By leveraging such compact input units of higher information density, the LLM can access to a vast scope of context even with a small context window. Extensible embedding is systematically optimized in architecture and training method, which leads to multiple advantages. 1) High flexibility of context extension, which flexibly supports ad-hoc extension of diverse context lengths. 2) Strong sample efficiency of training, which enables the embedding model to be learned in a cost-effective way. 3) Superior compatibility with the existing LLMs, where the extensible embedding can be seamlessly introduced as a plug-in component. Comprehensive evaluations on long-context language modeling and understanding tasks verify extensible embedding as an effective, efficient, flexible, and compatible method to extend the LLM's context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11571v1">Ain't Misbehavin' -- Using LLMs to Generate Expressive Robot Behavior in Conversations with the Tabletop Robot Haru</a></div>
    <div class="paper-meta">
      📅 2024-02-18
      | 💬 Accepted as Late Breaking Report (LBR) at the 19th Annual ACM/IEEE International Conference on Human Robot Interaction (HRI '24)
    </div>
    <details class="paper-abstract">
      Social robots aim to establish long-term bonds with humans through engaging conversation. However, traditional conversational approaches, reliant on scripted interactions, often fall short in maintaining engaging conversations. This paper addresses this limitation by integrating large language models (LLMs) into social robots to achieve more dynamic and expressive conversations. We introduce a fully-automated conversation system that leverages LLMs to generate robot responses with expressive behaviors, congruent with the robot's personality. We incorporate robot behavior with two modalities: 1) a text-to-speech (TTS) engine capable of various delivery styles, and 2) a library of physical actions for the robot. We develop a custom, state-of-the-art emotion recognition model to dynamically select the robot's tone of voice and utilize emojis from LLM output as cues for generating robot actions. A demo of our system is available here. To illuminate design and implementation issues, we conduct a pilot study where volunteers chat with a social robot using our proposed system, and we analyze their feedback, conducting a rigorous error analysis of chat transcripts. Feedback was overwhelmingly positive, with participants commenting on the robot's empathy, helpfulness, naturalness, and entertainment. Most negative feedback was due to automatic speech recognition (ASR) errors which had limited impact on conversations. However, we observed a small class of errors, such as the LLM repeating itself or hallucinating fictitious information and human responses, that have the potential to derail conversations, raising important issues for LLM application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.11729v3">OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples</a></div>
    <div class="paper-meta">
      📅 2024-02-18
      | 💬 AAAI 2024 camera ready. Code and dataset available at https://github.com/ryuryukke/OUTFOX
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12408v1">ModelGPT: Unleashing LLM's Capabilities for Tailored Model Generation</a></div>
    <div class="paper-meta">
      📅 2024-02-18
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has revolutionized various sectors by automating routine tasks, marking a step toward the realization of Artificial General Intelligence (AGI). However, they still struggle to accommodate the diverse and specific needs of users and simplify the utilization of AI models for the average user. In response, we propose ModelGPT, a novel framework designed to determine and generate AI models specifically tailored to the data or task descriptions provided by the user, leveraging the capabilities of LLMs. Given user requirements, ModelGPT is able to provide tailored models at most 270x faster than the previous paradigms (e.g. all-parameter or LoRA finetuning). Comprehensive experiments on NLP, CV, and Tabular datasets attest to the effectiveness of our framework in making AI models more accessible and user-friendly. Our code is available at https://github.com/IshiKura-a/ModelGPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11443v1">Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-02-18
    </div>
    <details class="paper-abstract">
      This paper presents a benchmark self-evolving framework to dynamically evaluate rapidly advancing Large Language Models (LLMs), aiming for a more accurate assessment of their capabilities and limitations. We utilize a multi-agent system to manipulate the context or question of original instances, reframing new evolving instances with high confidence that dynamically extend existing benchmarks. Towards a more scalable, robust and fine-grained evaluation, we implement six reframing operations to construct evolving instances testing LLMs against diverse queries, data noise and probing their problem-solving sub-abilities. With this framework, we extend benchmark datasets of four tasks. Experimental results show a general performance decline in most LLMs against their original results. This decline under our scalable and robust evaluations, alongside our fine-grained evaluation, more accurately reflect models' capabilities. Besides, our framework widens performance discrepancies both between different models and within the same model across various tasks, facilitating more informed model selection for specific tasks (Code and data are available at https://github.com/NanshineLoong/Self-Evolving-Benchmark).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14833v1">CliqueParcel: An Approach For Batching LLM Prompts That Jointly Optimizes Efficiency And Faithfulness</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become pivotal in recent research. However, during the inference process, LLMs still require substantial resources. In this paper, we propose CliqueParcel, a method designed to improve the efficiency of LLMs via prompt batching. Existing strategies to optimize inference efficiency often compromise on output quality, leading to a discounted output problem. This issue might result in reduced accuracy or outputs that are less detailed. CliqueParcel is our answer to this challenge. While ensuring accuracy and minimizing deviations from the original outputs (i.e., faithfulness), our method significantly improves efficiency during inference. To lay the groundwork, we first redefine efficiency measurements by excluding the reduction in running time due to shorter lengths. Then, we provide a comprehensive trade-off between efficiency and faithfulness to clarify the nature of the 'discounted output' problem. Within the CliqueParcel framework, we suggest multiple batching sub-methods and discuss the specific scenarios in which they can be applied. During evaluation, CliqueParcel is tested on eight widely recognized datasets, which can be classified into three types: reading comprehension, open-source question-answering, and reasoning. Our experiments explore the performance of CliqueParcel, including efficiency, faithfulness, and the trade-off between them. This work provides novel insights into inference efficiency and demonstrates promising performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11296v1">Dissecting Human and LLM Preferences</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      As a relative quality comparison of model responses, human and Large Language Model (LLM) preferences serve as common alignment goals in model fine-tuning and criteria in evaluation. Yet, these preferences merely reflect broad tendencies, resulting in less explainable and controllable models with potential safety risks. In this work, we dissect the preferences of human and 32 different LLMs to understand their quantitative composition, using annotations from real-world user-model conversations for a fine-grained, scenario-wise analysis. We find that humans are less sensitive to errors, favor responses that support their stances, and show clear dislike when models admit their limits. On the contrary, advanced LLMs like GPT-4-Turbo emphasize correctness, clarity, and harmlessness more. Additionally, LLMs of similar sizes tend to exhibit similar preferences, regardless of their training methods, and fine-tuning for alignment does not significantly alter the preferences of pretrained-only LLMs. Finally, we show that preference-based evaluation can be intentionally manipulated. In both training-free and training-based settings, aligning a model with the preferences of judges boosts scores, while injecting the least preferred properties lowers them. This results in notable score shifts: up to 0.59 on MT-Bench (1-10 scale) and 31.94 on AlpacaEval 2.0 (0-100 scale), highlighting the significant impact of this strategic adaptation. Interactive Demo: https://huggingface.co/spaces/GAIR/Preference-Dissection-Visualization Dataset: https://huggingface.co/datasets/GAIR/preference-dissection Code: https://github.com/GAIR-NLP/Preference-Dissection
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11260v1">MoRAL: MoE Augmented LoRA for LLMs' Lifelong Learning</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to new domains/tasks and enabling them to be efficient lifelong learners is a pivotal challenge. In this paper, we propose MoRAL, i.e., Mixture-of-Experts augmented Low-Rank Adaptation for Lifelong Learning. MoRAL combines the multi-tasking abilities of MoE with the fine-tuning abilities of LoRA for effective life-long learning of LLMs. In contrast to the conventional approaches that use factual triplets as inputs MoRAL relies on simple question-answer pairs, which is a more practical and effective strategy for robust and efficient learning. Owing to new data settings, we introduce a new evaluation benchmark namely: Life Long Learning of LLM (5L-bench) encompassing a newly curated dataset of question-answer pairs, and a set of evaluation metrics for rigorous evaluation of MoRAL in open-book and closed-book settings. Experimental evaluation shows (i) LLMs learn fast in open-book settings with up to 30.15% improvement in "RA" for Phi-2-2.7B compared to closed-book (for models fine-tuned with MoRAL); (ii) MoRAL shows higher performance improvement for models with a greater number of parameters; (iii) MoRAL is robust to catastrophic forgetting offering better knowledge retention compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11251v1">LLM can Achieve Self-Regulation via Hyperparameter Aware Generation</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      In the realm of Large Language Models (LLMs), users commonly employ diverse decoding strategies and adjust hyperparameters to control the generated text. However, a critical question emerges: Are LLMs conscious of the existence of these decoding strategies and capable of regulating themselves? The current decoding generation process often relies on empirical and heuristic manual adjustments to hyperparameters based on types of tasks and demands. However, this process is typically cumbersome, and the decoding hyperparameters may not always be optimal for each sample. To address the aforementioned challenges, we propose a novel text generation paradigm termed Hyperparameter Aware Generation (HAG). By leveraging hyperparameter-aware instruction tuning, the LLM autonomously determines the optimal decoding strategy and configs based on the input samples, enabling self-regulation. Our approach eliminates the need for extensive manual tuning, offering a more autonomous, self-regulate model behavior. Experimental results spanning six datasets across reasoning, creativity, translation, and mathematics tasks demonstrate that hyperparameter-aware instruction tuning empowers the LLMs to self-regulate the decoding strategy and hyperparameter. HAG extends the current paradigm in the text generation process, highlighting the feasibility of endowing the LLMs with self-regulate decoding strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11190v1">Disclosure and Mitigation of Gender Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-17
      | 💬 The first two authors contribute equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate biased responses. Yet previous direct probing techniques contain either gender mentions or predefined gender stereotypes, which are challenging to comprehensively collect. Hence, we propose an indirect probing framework based on conditional generation. This approach aims to induce LLMs to disclose their gender bias even without explicit gender or stereotype mentions. We explore three distinct strategies to disclose explicit and implicit gender bias in LLMs. Our experiments demonstrate that all tested LLMs exhibit explicit and/or implicit gender bias, even when gender stereotypes are not present in the inputs. In addition, an increased model size or model alignment amplifies bias in most cases. Furthermore, we investigate three methods to mitigate bias in LLMs via Hyperparameter Tuning, Instruction Guiding, and Debias Tuning. Remarkably, these methods prove effective even in the absence of explicit genders or stereotypes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16578v3">Leveraging Professional Radiologists' Expertise to Enhance LLMs' Evaluation for Radiology Reports</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      In radiology, Artificial Intelligence (AI) has significantly advanced report generation, but automatic evaluation of these AI-produced reports remains challenging. Current metrics, such as Conventional Natural Language Generation (NLG) and Clinical Efficacy (CE), often fall short in capturing the semantic intricacies of clinical contexts or overemphasize clinical details, undermining report clarity. To overcome these issues, our proposed method synergizes the expertise of professional radiologists with Large Language Models (LLMs), like GPT-3.5 and GPT-4 1. Utilizing In-Context Instruction Learning (ICIL) and Chain of Thought (CoT) reasoning, our approach aligns LLM evaluations with radiologist standards, enabling detailed comparisons between human and AI generated reports. This is further enhanced by a Regression model that aggregates sentence evaluation scores. Experimental results show that our "Detailed GPT-4 (5-shot)" model achieves a 0.48 score, outperforming the METEOR metric by 0.19, while our "Regressed GPT-4" model shows even greater alignment with expert evaluations, exceeding the best existing metric by a 0.35 margin. Moreover, the robustness of our explanations has been validated through a thorough iterative strategy. We plan to publicly release annotations from radiology experts, setting a new standard for accuracy in future assessments. This underscores the potential of our approach in enhancing the quality assessment of AI-driven medical reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09959v2">LLM-based Federated Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-02-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), with their advanced contextual understanding abilities, have demonstrated considerable potential in enhancing recommendation systems via fine-tuning methods. However, fine-tuning requires users' behavior data, which poses considerable privacy risks due to the incorporation of sensitive user information. The unintended disclosure of such data could infringe upon data protection laws and give rise to ethical issues. To mitigate these privacy issues, Federated Learning for Recommendation (Fed4Rec) has emerged as a promising approach. Nevertheless, applying Fed4Rec to LLM-based recommendation presents two main challenges: first, an increase in the imbalance of performance across clients, affecting the system's efficiency over time, and second, a high demand on clients' computational and storage resources for local training and inference of LLMs. To address these challenges, we introduce a Privacy-Preserving LLM-based Recommendation (PPLR) framework. The PPLR framework employs two primary strategies. First, it implements a dynamic balance strategy, which involves the design of dynamic parameter aggregation and adjustment of learning speed for different clients during the training phase, to ensure relatively balanced performance across all clients. Second, PPLR adopts a flexible storage strategy, selectively retaining certain sensitive layers of the language model on the client side while offloading non-sensitive layers to the server. This approach aims to preserve user privacy while efficiently saving computational and storage resources. Experimental results demonstrate that PPLR not only achieves a balanced performance among clients but also enhances overall system performance in a manner that is both computationally and storage-efficient, while effectively protecting user privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.02277v2">Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      We present Junk DNA Hypothesis by adopting a novel task-centric angle for the pre-trained weights of large language models (LLMs). It has been believed that weights in LLMs contain significant redundancy, leading to the conception that a considerable chunk of the parameters can be removed by pruning without compromising performance. Contrary to this belief, this paper presents a counter-argument: small-magnitude weights of pre-trained model weights encode vital knowledge essential for tackling difficult downstream tasks - manifested as the monotonic relationship between the performance drop of downstream tasks across the difficulty spectrum, as we prune more pre-trained weights by magnitude. Moreover, we reveal that these seemingly inconsequential weights can result in irreparable loss of knowledge and performance degradation in difficult tasks, even when downstream continual training is allowed. Interestingly, our evaluations show that the other popular compression, namely quantization, fails to exhibit similar monotonic effect and does not as convincingly disentangle this task-difficulty information. To study formally, we introduce several quantifiable metrics to gauge the downstream task difficulty: (1) within the same task category, and (2) across different task categories. Our extensive experiments substantiate the Junk DNA Hypothesis across a diverse range of model sizes, tasks, datasets, and even pruning methods. Codes are available at: https://github.com/VITA-Group/Junk_DNA_Hypothesis.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.00867v3">Do Compressed LLMs Forget Knowledge? An Experimental Study with Practical Implications</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      Compressing Large Language Models (LLMs) often leads to reduced performance, especially for knowledge-intensive tasks. In this work, we dive into how compression damages LLMs' inherent knowledge and the possible remedies. We start by proposing two conjectures on the nature of the damage: one is certain knowledge being forgotten (or erased) after LLM compression, hence necessitating the compressed model to (re)learn from data with additional parameters; the other presumes that knowledge is internally displaced and hence one requires merely "inference re-direction" with input-side augmentation such as prompting, to recover the knowledge-related performance. Extensive experiments are then designed to (in)validate the two conjectures. We observe the promise of prompting in comparison to model tuning; we further unlock prompting's potential by introducing a variant called Inference-time Dynamic Prompting (IDP), that can effectively increase prompt diversity without incurring any inference overhead. Our experiments consistently suggest that compared to the classical re-training alternatives such as LoRA, prompting with IDP leads to better or comparable post-compression performance recovery, while saving the extra parameter size by 21x and reducing inference latency by 60%. Our experiments hence strongly endorse the conjecture of "knowledge displaced" over "knowledge forgotten", and shed light on a new efficient mechanism to restore compressed LLM performance. We additionally visualize and analyze the different attention and activation patterns between prompted and re-trained models, demonstrating they achieve performance recovery in two different regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09731v2">Examining LLMs' Uncertainty Expression Towards Questions Outside Parametric Knowledge</a></div>
    <div class="paper-meta">
      📅 2024-02-16
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) express their uncertainty in situations where they lack sufficient parametric knowledge to generate reasonable responses? This work aims to systematically investigate LLMs' behaviors in such situations, emphasizing the trade-off between honesty and helpfulness. To tackle the challenge of precisely determining LLMs' knowledge gaps, we diagnostically create unanswerable questions containing non-existent concepts or false premises, ensuring that they are outside the LLMs' vast training data. By compiling a benchmark, UnknownBench, which consists of both unanswerable and answerable questions, we quantitatively evaluate the LLMs' performance in maintaining honesty while being helpful. Using a model-agnostic unified confidence elicitation approach, we observe that most LLMs fail to consistently refuse or express uncertainty towards questions outside their parametric knowledge, although instruction fine-tuning and alignment techniques can provide marginal enhancements. Moreover, LLMs' uncertainty expression does not always stay consistent with the perceived confidence of their textual outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.09665v2">The Wisdom of Partisan Crowds: Comparing Collective Intelligence in Humans and LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      Human groups are able to converge on more accurate beliefs through deliberation, even in the presence of polarization and partisan bias -- a phenomenon known as the "wisdom of partisan crowds." Generated agents powered by Large Language Models (LLMs) are increasingly used to simulate human collective behavior, yet few benchmarks exist for evaluating their dynamics against the behavior of human groups. In this paper, we examine the extent to which the wisdom of partisan crowds emerges in groups of LLM-based agents that are prompted to role-play as partisan personas (e.g., Democrat or Republican). We find that they not only display human-like partisan biases, but also converge to more accurate beliefs through deliberation as humans do. We then identify several factors that interfere with convergence, including the use of chain-of-thought prompt and lack of details in personas. Conversely, fine-tuning on human data appears to enhance convergence. These findings show the potential and limitations of LLM-based agents as a model of human collective intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10787v1">EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge</a></div>
    <div class="paper-meta">
      📅 2024-02-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Despite the remarkable strides of Large Language Models (LLMs) in various fields, the wide applications of LLMs on edge devices are limited due to their massive parameters and computations. To address this, quantization is commonly adopted to generate lightweight LLMs with efficient computations and fast inference. However, Post-Training Quantization (PTQ) methods dramatically degrade in quality when quantizing weights, activations, and KV cache together to below 8 bits. Besides, many Quantization-Aware Training (QAT) works quantize model weights, leaving the activations untouched, which do not fully exploit the potential of quantization for inference acceleration on the edge. In this paper, we propose EdgeQAT, the Entropy and Distribution Guided QAT for the optimization of lightweight LLMs to achieve inference acceleration on Edge devices. We first identify that the performance drop of quantization primarily stems from the information distortion in quantized attention maps, demonstrated by the different distributions in quantized query and key of the self-attention mechanism. Then, the entropy and distribution guided QAT is proposed to mitigate the information distortion. Moreover, we design a token importance-aware adaptive method to dynamically quantize the tokens with different bit widths for further optimization and acceleration. Our extensive experiments verify the substantial improvements with our framework across various datasets. Furthermore, we achieve an on-device speedup of up to 2.37x compared with its FP16 counterparts across multiple edge devices, signaling a groundbreaking advancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07470v2">Pushing The Limit of LLM Capacity for Text Classification</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      The value of text classification's future research has encountered challenges and uncertainties, due to the extraordinary efficacy demonstrated by large language models (LLMs) across numerous downstream NLP tasks. In this era of open-ended language modeling, where task boundaries are gradually fading, an urgent question emerges: have we made significant advances in text classification under the full benefit of LLMs? To answer this question, we propose RGPT, an adaptive boosting framework tailored to produce a specialized text classification LLM by recurrently ensembling a pool of strong base learners. The base learners are constructed by adaptively adjusting the distribution of training samples and iteratively fine-tuning LLMs with them. Such base learners are then ensembled to be a specialized text classification LLM, by recurrently incorporating the historical predictions from the previous learners. Through a comprehensive empirical comparison, we show that RGPT significantly outperforms 8 SOTA PLMs and 7 SOTA LLMs on four benchmarks by 1.36% on average. Further evaluation experiments show a clear surpassing of RGPT over human classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16240v2">Combining Hierachical VAEs with LLMs for clinically meaningful timeline summarisation in social media</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      We introduce a hybrid abstractive summarisation approach combining hierarchical VAE with LLMs (LlaMA-2) to produce clinically meaningful summaries from social media user timelines, appropriate for mental health monitoring. The summaries combine two different narrative points of view: clinical insights in third person useful for a clinician are generated by feeding into an LLM specialised clinical prompts, and importantly, a temporally sensitive abstractive summary of the user's timeline in first person, generated by a novel hierarchical variational autoencoder, TH-VAE. We assess the generated summaries via automatic evaluation against expert summaries and via human evaluation with clinical experts, showing that timeline summarisation by TH-VAE results in more factual and logically coherent summaries rich in clinical utility and superior to LLM-only approaches in capturing changes over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.07324v3">Small LLMs Are Weak Tool Learners: A Multi-LLM Agent</a></div>
    <div class="paper-meta">
      📅 2024-02-16
      | 💬 On progress, github repo: https://github.com/X-PLUG/Multi-LLM-Agent
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents significantly extend the capabilities of standalone LLMs, empowering them to interact with external tools (e.g., APIs, functions) and complete various tasks in a self-directed fashion. The challenge of tool use demands that LLMs not only understand user queries and generate answers accurately but also excel in task planning, tool invocation, and result summarization. While traditional works focus on training a single LLM with all these capabilities, performance limitations become apparent, particularly with smaller models. To overcome these challenges, we propose a novel approach that decomposes the aforementioned capabilities into a planner, caller, and summarizer. Each component is implemented by a single LLM that focuses on a specific capability and collaborates with others to accomplish the task. This modular framework facilitates individual updates and the potential use of smaller LLMs for building each capability. To effectively train this framework, we introduce a two-stage training paradigm. First, we fine-tune a backbone LLM on the entire dataset without discriminating sub-tasks, providing the model with a comprehensive understanding of the task. Second, the fine-tuned LLM is used to instantiate the planner, caller, and summarizer respectively, which are continually fine-tuned on respective sub-tasks. Evaluation across various tool-use benchmarks illustrates that our proposed multi-LLM framework surpasses the traditional single-LLM approach, highlighting its efficacy and advantages in tool learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10631v1">BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      The upscaling of Large Language Models (LLMs) has yielded impressive advances in natural language processing, yet it also poses significant deployment challenges. Weight quantization has emerged as a widely embraced solution to reduce memory and computational demands. This paper introduces BitDistiller, a framework that synergizes Quantization-Aware Training (QAT) with Knowledge Distillation (KD) to boost the performance of LLMs at ultra-low precisions (sub-4-bit). Specifically, BitDistiller first incorporates a tailored asymmetric quantization and clipping technique to maximally preserve the fidelity of quantized weights, and then proposes a novel Confidence-Aware Kullback-Leibler Divergence (CAKLD) objective, which is employed in a self-distillation manner to enable faster convergence and superior model performance. Empirical evaluations demonstrate that BitDistiller significantly surpasses existing methods in both 3-bit and 2-bit configurations on general language understanding and complex reasoning benchmarks. Notably, BitDistiller is shown to be more cost-effective, demanding fewer data and training resources. The code is available at https://github.com/DD-DuDa/BitDistiller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09091v2">Play Guessing Game with LLM: Indirect Jailbreak Attack with Implicit Clues</a></div>
    <div class="paper-meta">
      📅 2024-02-16
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      With the development of LLMs, the security threats of LLMs are getting more and more attention. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks primarily utilize scenario camouflage techniques. However their explicitly mention of malicious intent will be easily recognized and defended by LLMs. In this paper, we propose an indirect jailbreak attack approach, Puzzler, which can bypass the LLM's defense strategy and obtain malicious response by implicitly providing LLMs with some clues about the original malicious query. In addition, inspired by the wisdom of "When unable to attack, defend" from Sun Tzu's Art of War, we adopt a defensive stance to gather clues about the original malicious query through LLMs. Extensive experimental results show that Puzzler achieves a query success rate of 96.6% on closed-source LLMs, which is 57.9%-82.7% higher than baselines. Furthermore, when tested against the state-of-the-art jailbreak detection approaches, Puzzler proves to be more effective at evading detection compared to baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10532v1">Properties and Challenges of LLM-Generated Explanations</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      The self-rationalising capabilities of large language models (LLMs) have been explored in restricted settings, using task/specific data sets. However, current LLMs do not (only) rely on specifically annotated data; nonetheless, they frequently explain their outputs. The properties of the generated explanations are influenced by the pre-training corpus and by the target data used for instruction fine-tuning. As the pre-training corpus includes a large amount of human-written explanations "in the wild", we hypothesise that LLMs adopt common properties of human explanations. By analysing the outputs for a multi-domain instruction fine-tuning data set, we find that generated explanations show selectivity and contain illustrative elements, but less frequently are subjective or misleading. We discuss reasons and consequences of the properties' presence or absence. In particular, we outline positive and negative implications depending on the goals and user groups of the self-rationalising system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10524v1">LLM Comparator: Visual Analytics for Side-by-Side Evaluation of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      Automatic side-by-side evaluation has emerged as a promising approach to evaluating the quality of responses from large language models (LLMs). However, analyzing the results from this evaluation approach raises scalability and interpretability challenges. In this paper, we present LLM Comparator, a novel visual analytics tool for interactively analyzing results from automatic side-by-side evaluation. The tool supports interactive workflows for users to understand when and why a model performs better or worse than a baseline model, and how the responses from two models are qualitatively different. We iteratively designed and developed the tool by closely working with researchers and engineers at a large technology company. This paper details the user challenges we identified, the design and development of the tool, and an observational study with participants who regularly evaluate their models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06664v3">LLM Agents can Autonomously Hack Websites</a></div>
    <div class="paper-meta">
      📅 2024-02-16
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have become increasingly capable and can now interact with tools (i.e., call functions), read documents, and recursively call themselves. As a result, these LLMs can now function autonomously as agents. With the rise in capabilities of these agents, recent work has speculated on how LLM agents would affect cybersecurity. However, not much is known about the offensive capabilities of LLM agents. In this work, we show that LLM agents can autonomously hack websites, performing tasks as complex as blind database schema extraction and SQL injections without human feedback. Importantly, the agent does not need to know the vulnerability beforehand. This capability is uniquely enabled by frontier models that are highly capable of tool use and leveraging extended context. Namely, we show that GPT-4 is capable of such hacks, but existing open-source models are not. Finally, we show that GPT-4 is capable of autonomously finding vulnerabilities in websites in the wild. Our findings raise questions about the widespread deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.09733v1">Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States</a></div>
    <div class="paper-meta">
      📅 2024-02-16
      | 💬 9 pages, 8 figures, 2 tables (13 pages, 12 figures, 13 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can make up answers that are not real, and this is known as hallucination. This research aims to see if, how, and to what extent LLMs are aware of hallucination. More specifically, we check whether and how an LLM reacts differently in its hidden states when it answers a question right versus when it hallucinates. To do this, we introduce an experimental framework which allows examining LLM's hidden states in different hallucination situations. Building upon this framework, we conduct a series of experiments with language models in the LLaMA family (Touvron et al., 2023). Our empirical findings suggest that LLMs react differently when processing a genuine response versus a fabricated one. We then apply various model interpretation techniques to help understand and explain the findings better. Moreover, informed by the empirical observations, we show great potential of using the guidance derived from LLM's hidden representation space to mitigate hallucination. We believe this work provides insights into how LLMs produce hallucinated answers and how to make them occur less often.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10294v1">LAVE: LLM-Powered Agent Assistance and Language Augmentation for Video Editing</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 Paper accepted to the ACM Conference on Intelligent User Interfaces (ACM IUI) 2024
    </div>
    <details class="paper-abstract">
      Video creation has become increasingly popular, yet the expertise and effort required for editing often pose barriers to beginners. In this paper, we explore the integration of large language models (LLMs) into the video editing workflow to reduce these barriers. Our design vision is embodied in LAVE, a novel system that provides LLM-powered agent assistance and language-augmented editing features. LAVE automatically generates language descriptions for the user's footage, serving as the foundation for enabling the LLM to process videos and assist in editing tasks. When the user provides editing objectives, the agent plans and executes relevant actions to fulfill them. Moreover, LAVE allows users to edit videos through either the agent or direct UI manipulation, providing flexibility and enabling manual refinement of agent actions. Our user study, which included eight participants ranging from novices to proficient editors, demonstrated LAVE's effectiveness. The results also shed light on user perceptions of the proposed LLM-assisted editing paradigm and its impact on users' creativity and sense of co-creation. Based on these findings, we propose design implications to inform the future development of agent-assisted content editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10083v1">Fine-tuning Large Language Model (LLM) Artificial Intelligence Chatbots in Ophthalmology and LLM-based evaluation using GPT-4</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 13 Pages, 1 Figure, 8 Tables
    </div>
    <details class="paper-abstract">
      Purpose: To assess the alignment of GPT-4-based evaluation to human clinician experts, for the evaluation of responses to ophthalmology-related patient queries generated by fine-tuned LLM chatbots. Methods: 400 ophthalmology questions and paired answers were created by ophthalmologists to represent commonly asked patient questions, divided into fine-tuning (368; 92%), and testing (40; 8%). We find-tuned 5 different LLMs, including LLAMA2-7b, LLAMA2-7b-Chat, LLAMA2-13b, and LLAMA2-13b-Chat. For the testing dataset, additional 8 glaucoma QnA pairs were included. 200 responses to the testing dataset were generated by 5 fine-tuned LLMs for evaluation. A customized clinical evaluation rubric was used to guide GPT-4 evaluation, grounded on clinical accuracy, relevance, patient safety, and ease of understanding. GPT-4 evaluation was then compared against ranking by 5 clinicians for clinical alignment. Results: Among all fine-tuned LLMs, GPT-3.5 scored the highest (87.1%), followed by LLAMA2-13b (80.9%), LLAMA2-13b-chat (75.5%), LLAMA2-7b-Chat (70%) and LLAMA2-7b (68.8%) based on the GPT-4 evaluation. GPT-4 evaluation demonstrated significant agreement with human clinician rankings, with Spearman and Kendall Tau correlation coefficients of 0.90 and 0.80 respectively; while correlation based on Cohen Kappa was more modest at 0.50. Notably, qualitative analysis and the glaucoma sub-analysis revealed clinical inaccuracies in the LLM-generated responses, which were appropriately identified by the GPT-4 evaluation. Conclusion: The notable clinical alignment of GPT-4 evaluation highlighted its potential to streamline the clinical evaluation of LLM chatbot responses to healthcare-related queries. By complementing the existing clinician-dependent manual grading, this efficient and automated evaluation could assist the validation of future developments in LLM applications for healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10076v1">QUICK: Quantization-aware Interleaving and Conflict-free Kernel for efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 9 pages, 8 figures
    </div>
    <details class="paper-abstract">
      We introduce QUICK, a group of novel optimized CUDA kernels for the efficient inference of quantized Large Language Models (LLMs). QUICK addresses the shared memory bank-conflict problem of state-of-the-art mixed precision matrix multiplication kernels. Our method interleaves the quantized weight matrices of LLMs offline to skip the shared memory write-back after the dequantization. We demonstrate up to 1.91x speedup over existing kernels of AutoAWQ on larger batches and up to 1.94x throughput gain on representative LLM models on various NVIDIA GPU devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10051v1">SwissNYF: Tool Grounded LLM Agents for Black Box Setting</a></div>
    <div class="paper-meta">
      📅 2024-02-15
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated enhanced capabilities in function-calling, these advancements primarily rely on accessing the functions' responses. This methodology is practical for simpler APIs but faces scalability issues with irreversible APIs that significantly impact the system, such as a database deletion API. Similarly, processes requiring extensive time for each API call and those necessitating forward planning, like automated action pipelines, present complex challenges. Furthermore, scenarios often arise where a generalized approach is needed because algorithms lack direct access to the specific implementations of these functions or secrets to use them. Traditional tool planning methods are inadequate in these cases, compelling the need to operate within black-box environments. Unlike their performance in tool manipulation, LLMs excel in black-box tasks, such as program synthesis. Therefore, we harness the program synthesis capabilities of LLMs to strategize tool usage in black-box settings, ensuring solutions are verified prior to implementation. We introduce TOPGUN, an ingeniously crafted approach leveraging program synthesis for black box tool planning. Accompanied by SwissNYF, a comprehensive suite that integrates black-box algorithms for planning and verification tasks, addressing the aforementioned challenges and enhancing the versatility and effectiveness of LLMs in complex API interactions. The public code for SwissNYF is available at https://github.com/iclr-dummy-user/SwissNYF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.03662v1">X-lifecycle Learning for Cloud Incident Management using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-15
    </div>
    <details class="paper-abstract">
      Incident management for large cloud services is a complex and tedious process and requires significant amount of manual efforts from on-call engineers (OCEs). OCEs typically leverage data from different stages of the software development lifecycle [SDLC] (e.g., codes, configuration, monitor data, service properties, service dependencies, trouble-shooting documents, etc.) to generate insights for detection, root causing and mitigating of incidents. Recent advancements in large language models [LLMs] (e.g., ChatGPT, GPT-4, Gemini) created opportunities to automatically generate contextual recommendations to the OCEs assisting them to quickly identify and mitigate critical issues. However, existing research typically takes a silo-ed view for solving a certain task in incident management by leveraging data from a single stage of SDLC. In this paper, we demonstrate that augmenting additional contextual data from different stages of SDLC improves the performance of two critically important and practically challenging tasks: (1) automatically generating root cause recommendations for dependency failure related incidents, and (2) identifying ontology of service monitors used for automatically detecting incidents. By leveraging 353 incident and 260 monitor dataset from Microsoft, we demonstrate that augmenting contextual information from different stages of the SDLC improves the performance over State-of-The-Art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09733v1">Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 9 pages, 8 figures, 2 tables (13 pages, 12 figures, 13 tables including references and appendices)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can make up answers that are not real, and this is known as hallucination. This research aims to see if, how, and to what extent LLMs are aware of hallucination. More specifically, we check whether and how an LLM reacts differently in its hidden states when it answers a question right versus when it hallucinates. To do this, we introduce an experimental framework which allows examining LLM's hidden states in different hallucination situations. Building upon this framework, we conduct a series of experiments with language models in the LLaMA family (Touvron et al., 2023). Our empirical findings suggest that LLMs react differently when processing a genuine response versus a fabricated one. We then apply various model interpretation techniques to help understand and explain the findings better. Moreover, informed by the empirical observations, we show great potential of using the guidance derived from LLM's hidden representation space to mitigate hallucination. We believe this work provides insights into how LLMs produce hallucinated answers and how to make them occur less often.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.08901v3">Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning</a></div>
    <div class="paper-meta">
      📅 2024-02-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive capabilities, yet they still struggle with math reasoning. In this work, we propose CoT-Influx, a novel approach that pushes the boundary of few-shot Chain-of-Thoughts (CoT) learning to improve LLM mathematical reasoning. Motivated by the observation that adding more concise CoT examples in the prompt can improve LLM reasoning performance, CoT-Influx employs a coarse-to-fine pruner to maximize the input of effective and concise CoT examples. The pruner first selects as many crucial CoT examples as possible and then prunes unimportant tokens to fit the context window. A math reasoning dataset with diverse difficulty levels and reasoning steps is used to train the pruner, along with a math-specialized reinforcement learning approach. As a result, by enabling more CoT examples with double the context window size in tokens, CoT-Influx significantly outperforms various prompting baselines across various LLMs (LLaMA2-7B, 13B, 70B) and 5 math datasets, achieving up to 4.55% absolute improvements. Remarkably, without any fine-tuning, LLaMA2-70B with CoT-Influx surpasses GPT-3.5 and a wide range of larger LLMs (PaLM, Minerva 540B, etc.) on the GSM8K. CoT-Influx serves as a plug-and-play module for LLMs and is compatible with most existing reasoning prompting techniques, such as self-consistency and self-verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09668v1">How to Train Data-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-15
      | 💬 Under review. 44 pages, 30 figures
    </div>
    <details class="paper-abstract">
      The training of large language models (LLMs) is expensive. In this paper, we study data-efficient approaches for pre-training LLMs, i.e., techniques that aim to optimize the Pareto frontier of model quality and training resource/data consumption. We seek to understand the tradeoffs associated with data selection routines based on (i) expensive-to-compute data-quality estimates, and (ii) maximization of coverage and diversity-based measures in the feature space. Our first technique, Ask-LLM, leverages the zero-shot reasoning capabilities of instruction-tuned LLMs to directly assess the quality of a training example. To target coverage, we propose Density sampling, which models the data distribution to select a diverse sample. In our comparison of 19 samplers, involving hundreds of evaluation tasks and pre-training runs, we find that Ask-LLM and Density are the best methods in their respective categories. Coverage sampling can recover the performance of the full data, while models trained on Ask-LLM data consistently outperform full-data training -- even when we reject 90% of the original dataset, while converging up to 70% faster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09617v1">LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations</a></div>
    <div class="paper-meta">
      📅 2024-02-14
    </div>
    <details class="paper-abstract">
      The extraordinary performance of large language models has not only reshaped the research landscape in the field of NLP but has also demonstrated its exceptional applicative potential in various domains. However, the potential of these models in mining relationships from graph data remains under-explored. Graph neural networks, as a popular research area in recent years, have numerous studies on relationship mining. Yet, current cutting-edge research in graph neural networks has not been effectively integrated with large language models, leading to limited efficiency and capability in graph relationship mining tasks. A primary challenge is the inability of LLMs to deeply exploit the edge information in graphs, which is critical for understanding complex node relationships. This gap limits the potential of LLMs to extract meaningful insights from graph structures, limiting their applicability in more complex graph-based analysis. We focus on how to utilize existing LLMs for mining and understanding relationships in graph data, applying these techniques to recommendation tasks. We propose an innovative framework that combines the strong contextual representation capabilities of LLMs with the relationship extraction and analysis functions of GNNs for mining relationships in graph data. Specifically, we design a new prompt construction framework that integrates relational information of graph data into natural language expressions, aiding LLMs in more intuitively grasping the connectivity information within graph data. Additionally, we introduce graph relationship understanding and analysis functions into LLMs to enhance their focus on connectivity information in graph data. Our evaluation on real-world datasets demonstrates the framework's ability to understand connectivity information in graph data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.18354v2">LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations</a></div>
    <div class="paper-meta">
      📅 2024-02-14
      | 💬 26 pages, 15 figures, published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      Can a Large Language Model (LLM) solve simple abstract reasoning problems? We explore this broad question through a systematic analysis of GPT on the Abstraction and Reasoning Corpus (ARC), a representative benchmark of abstract reasoning ability from limited examples in which solutions require some "core knowledge" of concepts such as objects, goal states, counting, and basic geometry. GPT-4 solves only 13/50 of the most straightforward ARC tasks when using textual encodings for their two-dimensional input-output grids. Our failure analysis reveals that GPT-4's capacity to identify objects and reason about them is significantly influenced by the sequential nature of the text that represents an object within a text encoding of a task. To test this hypothesis, we design a new benchmark, the 1D-ARC, which consists of one-dimensional (array-like) tasks that are more conducive to GPT-based reasoning, and where it indeed performs better than on the (2D) ARC. To alleviate this issue, we propose an object-based representation that is obtained through an external tool, resulting in nearly doubling the performance on solved ARC tasks and near-perfect scores on the easier 1D-ARC. Although the state-of-the-art GPT-4 is unable to "reason" perfectly within non-language domains such as the 1D-ARC or a simple ARC subset, our study reveals that the use of object-based representations can significantly improve its reasoning ability. Visualizations, GPT logs, and data are available at https://khalil-research.github.io/LLM4ARC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09546v1">How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?</a></div>
    <div class="paper-meta">
      📅 2024-02-14
    </div>
    <details class="paper-abstract">
      In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09404v1">AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability</a></div>
    <div class="paper-meta">
      📅 2024-02-14
    </div>
    <details class="paper-abstract">
      This paper introduces AQA-Bench, a novel benchmark to assess the sequential reasoning capabilities of large language models (LLMs) in algorithmic contexts, such as depth-first search (DFS). The key feature of our evaluation benchmark lies in its interactive evaluation protocol -- for example, in DFS, the availability of each node's connected edge is contingent upon the model's traversal to that node, thereby necessitating the LLM's ability to effectively remember visited nodes and strategize subsequent moves. We comprehensively build AQA-Bench with three different algorithms, namely binary search, depth-first search, and breadth-first search, and to evaluate the sequential reasoning ability of 12 different LLMs. Our investigations reveal several interesting findings: (1) Closed-source models like GPT-4 and Gemini generally show strong sequential reasoning ability, significantly outperforming open-source LLMs. (2) Naively providing interactive examples may inadvertently hurt few-shot performance. (3) A very limited number of predecessor steps following the optimal policy can substantially boost small models' performance. (4) The scaling correlation between performance and model size is not always significant, sometimes even showcasing an inverse trend. We hope our study can catalyze future work on advancing the understanding and enhancement of LLMs' capabilities in sequential reasoning. The code is available at https://github.com/UCSC-VLAA/AQA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09360v1">HiRE: High Recall Approximate Top-$k$ Estimation for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-02-14
    </div>
    <details class="paper-abstract">
      Autoregressive decoding with generative Large Language Models (LLMs) on accelerators (GPUs/TPUs) is often memory-bound where most of the time is spent on transferring model parameters from high bandwidth memory (HBM) to cache. On the other hand, recent works show that LLMs can maintain quality with significant sparsity/redundancy in the feedforward (FFN) layers by appropriately training the model to operate on a top-$k$ fraction of rows/columns (where $k \approx 0.05$), there by suggesting a way to reduce the transfer of model parameters, and hence latency. However, exploiting this sparsity for improving latency is hindered by the fact that identifying top rows/columns is data-dependent and is usually performed using full matrix operations, severely limiting potential gains. To address these issues, we introduce HiRE (High Recall Approximate Top-k Estimation). HiRE comprises of two novel components: (i) a compression scheme to cheaply predict top-$k$ rows/columns with high recall, followed by full computation restricted to the predicted subset, and (ii) DA-TOP-$k$: an efficient multi-device approximate top-$k$ operator. We demonstrate that on a one billion parameter model, HiRE applied to both the softmax as well as feedforward layers, achieves almost matching pretraining and downstream accuracy, and speeds up inference latency by $1.47\times$ on a single TPUv5e device.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06072v2">Chain of History: Learning and Forecasting with LLMs for Temporal Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2024-02-14
      | 💬 15 pages; typos corrected, references added
    </div>
    <details class="paper-abstract">
      Temporal Knowledge Graph Completion (TKGC) is a complex task involving the prediction of missing event links at future timestamps by leveraging established temporal structural knowledge. This paper aims to provide a comprehensive perspective on harnessing the advantages of Large Language Models (LLMs) for reasoning in temporal knowledge graphs, presenting an easily transferable pipeline. In terms of graph modality, we underscore the LLMs' prowess in discerning the structural information of pivotal nodes within the historical chain. As for the generation mode of the LLMs utilized for inference, we conduct an exhaustive exploration into the variances induced by a range of inherent factors in LLMs, with particular attention to the challenges in comprehending reverse logic. We adopt a parameter-efficient fine-tuning strategy to harmonize the LLMs with the task requirements, facilitating the learning of the key knowledge highlighted earlier. Comprehensive experiments are undertaken on several widely recognized datasets, revealing that our framework exceeds or parallels existing methods across numerous popular metrics. Additionally, we execute a substantial range of ablation experiments and draw comparisons with several advanced commercial LLMs, to investigate the crucial factors influencing LLMs' performance in structured temporal knowledge inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09063v1">Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space</a></div>
    <div class="paper-meta">
      📅 2024-02-14
      | 💬 Trigger Warning: the appendix contains LLM-generated text with violence and harassment
    </div>
    <details class="paper-abstract">
      Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08846v1">An Embarrassingly Simple Approach for LLM with Strong ASR Capacity</a></div>
    <div class="paper-meta">
      📅 2024-02-13
      | 💬 Working in progress and will open-source soon
    </div>
    <details class="paper-abstract">
      In this paper, we focus on solving one of the most important tasks in the field of speech processing, i.e., automatic speech recognition (ASR), with speech foundation encoders and large language models (LLM). Recent works have complex designs such as compressing the output temporally for the speech encoder, tackling modal alignment for the projector, and utilizing parameter-efficient fine-tuning for the LLM. We found that delicate designs are not necessary, while an embarrassingly simple composition of off-the-shelf speech encoder, LLM, and the only trainable linear projector is competent for the ASR task. To be more specific, we benchmark and explore various combinations of LLMs and speech encoders, leading to the optimal LLM-based ASR system, which we call SLAM-ASR. The proposed SLAM-ASR provides a clean setup and little task-specific design, where only the linear projector is trained. To the best of our knowledge, SLAM-ASR achieves the best performance on the Librispeech benchmark among LLM-based ASR models and even outperforms the latest LLM-based audio-universal model trained on massive pair data. Finally, we explore the capability emergence of LLM-based ASR in the process of modal alignment. We hope that our study can facilitate the research on extending LLM with cross-modality capacity and shed light on the LLM-based ASR community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08755v1">LLM-driven Imitation of Subrational Behavior : Illusion or Reality?</a></div>
    <div class="paper-meta">
      📅 2024-02-13
    </div>
    <details class="paper-abstract">
      Modeling subrational agents, such as humans or economic households, is inherently challenging due to the difficulty in calibrating reinforcement learning models or collecting data that involves human subjects. Existing work highlights the ability of Large Language Models (LLMs) to address complex reasoning tasks and mimic human communication, while simulation using LLMs as agents shows emergent social behaviors, potentially improving our comprehension of human conduct. In this paper, we propose to investigate the use of LLMs to generate synthetic human demonstrations, which are then used to learn subrational agent policies though Imitation Learning. We make an assumption that LLMs can be used as implicit computational models of humans, and propose a framework to use synthetic demonstrations derived from LLMs to model subrational behaviors that are characteristic of humans (e.g., myopic behavior or preference for risk aversion). We experimentally evaluate the ability of our framework to model sub-rationality through four simple scenarios, including the well-researched ultimatum game and marshmallow experiment. To gain confidence in our framework, we are able to replicate well-established findings from prior human studies associated with the above scenarios. We conclude by discussing the potential benefits, challenges and limitations of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15230v2">PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-13
      | 💬 27 pages, 4 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Neural Networks can be efficiently compressed through pruning, significantly reducing storage and computational demands while maintaining predictive performance. Simple yet effective methods like Iterative Magnitude Pruning (IMP, Han et al., 2015) remove less important parameters and require a costly retraining procedure to recover performance after pruning. However, with the rise of Large Language Models (LLMs), full retraining has become infeasible due to memory and compute constraints. In this study, we challenge the practice of retraining all parameters by demonstrating that updating only a small subset of highly expressive parameters is often sufficient to recover or even improve performance compared to full retraining. Surprisingly, retraining as little as 0.27%-0.35% of the parameters of GPT-architectures achieves comparable performance to One Shot IMP across various sparsity levels. Our approach, Parameter-Efficient Retraining after Pruning (PERP), drastically reduces compute and memory demands, enabling pruning and retraining of up to 30 billion parameter models on a single NVIDIA A100 GPU within minutes. Despite magnitude pruning being considered as unsuited for pruning LLMs, our findings show that PERP positions it as a strong contender against state-of-the-art retraining-free approaches such as Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023), opening up a promising alternative to avoiding retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15170v2">Scalable Qualitative Coding with LLMs: Chain-of-Thought Reasoning Matches Human Performance in Some Hermeneutic Tasks</a></div>
    <div class="paper-meta">
      📅 2024-02-12
    </div>
    <details class="paper-abstract">
      Qualitative coding, or content analysis, extracts meaning from text to discern quantitative patterns across a corpus of texts. Recently, advances in the interpretive abilities of large language models (LLMs) offer potential for automating the coding process (applying category labels to texts), thereby enabling human researchers to concentrate on more creative research aspects, while delegating these interpretive tasks to AI. Our case study comprises a set of socio-historical codes on dense, paragraph-long passages representative of a humanistic study. We show that GPT-4 is capable of human-equivalent interpretations, whereas GPT-3.5 is not. Compared to our human-derived gold standard, GPT-4 delivers excellent intercoder reliability (Cohen's $\kappa \geq 0.79$) for 3 of 9 codes, and substantial reliability ($\kappa \geq 0.6$) for 8 of 9 codes. In contrast, GPT-3.5 greatly underperforms for all codes ($mean(\kappa) = 0.34$; $max(\kappa) = 0.55$). Importantly, we find that coding fidelity improves considerably when the LLM is prompted to give rationale justifying its coding decisions (chain-of-thought reasoning). We present these and other findings along with a set of best practices for adapting traditional codebooks for LLMs. Our results indicate that for certain codebooks, state-of-the-art LLMs are already adept at large-scale content analysis. Furthermore, they suggest the next generation of models will likely render AI coding a viable option for a majority of codebooks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08064v1">Beyond LLMs: Advancing the Landscape of Complex Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-02-12
    </div>
    <details class="paper-abstract">
      Since the advent of Large Language Models a few years ago, they have often been considered the de facto solution for many AI problems. However, in addition to the many deficiencies of LLMs that prevent them from broad industry adoption, such as reliability, cost, and speed, there is a whole class of common real world problems that Large Language Models perform poorly on, namely, constraint satisfaction and optimization problems. These problems are ubiquitous and current solutions are highly specialized and expensive to implement. At Elemental Cognition, we developed our EC AI platform which takes a neuro-symbolic approach to solving constraint satisfaction and optimization problems. The platform employs, at its core, a precise and high performance logical reasoning engine, and leverages LLMs for knowledge acquisition and user interaction. This platform supports developers in specifying application logic in natural and concise language while generating application user interfaces to interact with users effectively. We evaluated LLMs against systems built on the EC AI platform in three domains and found the EC AI systems to significantly outperform LLMs on constructing valid and optimal solutions, on validating proposed solutions, and on repairing invalid solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.09342v3">Ranking LLM-Generated Loop Invariants for Program Verification</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 Findings of The 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP-findings 2023)
    </div>
    <details class="paper-abstract">
      Synthesizing inductive loop invariants is fundamental to automating program verification. In this work, we observe that Large Language Models (such as gpt-3.5 or gpt-4) are capable of synthesizing loop invariants for a class of programs in a 0-shot setting, yet require several samples to generate the correct invariants. This can lead to a large number of calls to a program verifier to establish an invariant. To address this issue, we propose a {\it re-ranking} approach for the generated results of LLMs. We have designed a ranker that can distinguish between correct inductive invariants and incorrect attempts based on the problem definition. The ranker is optimized as a contrastive ranker. Experimental results demonstrate that this re-ranking mechanism significantly improves the ranking of correct invariants among the generated candidates, leading to a notable reduction in the number of calls to a verifier. The source code and the experimental data for this paper are available in \url{https://github.com/microsoft/NeuralInvariantRanker}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08030v1">Why and When LLM-Based Assistants Can Go Wrong: Investigating the Effectiveness of Prompt-Based Interactions for Software Help-Seeking</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 Accepted for publication in the Proceedings of the 29th International Conference on Intelligent User Interfaces (IUI'24), March 18--21, 2024, in Greenville, SC, USA
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) assistants, such as ChatGPT, have emerged as potential alternatives to search methods for helping users navigate complex, feature-rich software. LLMs use vast training data from domain-specific texts, software manuals, and code repositories to mimic human-like interactions, offering tailored assistance, including step-by-step instructions. In this work, we investigated LLM-generated software guidance through a within-subject experiment with 16 participants and follow-up interviews. We compared a baseline LLM assistant with an LLM optimized for particular software contexts, SoftAIBot, which also offered guidelines for constructing appropriate prompts. We assessed task completion, perceived accuracy, relevance, and trust. Surprisingly, although SoftAIBot outperformed the baseline LLM, our results revealed no significant difference in LLM usage and user perceptions with or without prompt guidelines and the integration of domain context. Most users struggled to understand how the prompt's text related to the LLM's responses and often followed the LLM's suggestions verbatim, even if they were incorrect. This resulted in difficulties when using the LLM's advice for software tasks, leading to low task completion rates. Our detailed analysis also revealed that users remained unaware of inaccuracies in the LLM's responses, indicating a gap between their lack of software expertise and their ability to evaluate the LLM's assistance. With the growing push for designing domain-specific LLM assistants, we emphasize the importance of incorporating explainable, context-aware cues into LLMs to help users understand prompt-based interactions, identify biases, and maximize the utility of LLM assistants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.08005v1">Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 Pre-print. Submitted to the ICLR 2024 Workshop on Representational Alignment (Re-Align)
    </div>
    <details class="paper-abstract">
      In this paper, we introduce \emph{refined Direct Preference Optimization} (rDPO), a method for improving the behavioral alignment of Large Language Models (LLMs) without the need for human-annotated data. The method involves creating synthetic data using self-critique prompting by a teacher LLM and then utilising a generalized DPO loss function to distil to a student LLM. The loss function incorporates an additional external reward model to improve the quality of synthetic data, making rDPO robust to potential noise in the synthetic dataset. rDPO is shown to be effective in a diverse set of behavioural alignment tasks, such as improved safety, robustness against role-playing, and reduced sycophancy. Code to be released at https://github.com/vicgalle/refined-dpo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.08560v2">MemGPT: Towards LLMs as Operating Systems</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 Code and data available at https://research.memgpt.ai
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized AI, but are constrained by limited context windows, hindering their utility in tasks like extended conversations and document analysis. To enable using context beyond limited context windows, we propose virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems that provide the appearance of large memory resources through data movement between fast and slow memory. Using this technique, we introduce MemGPT (Memory-GPT), a system that intelligently manages different memory tiers in order to effectively provide extended context within the LLM's limited context window, and utilizes interrupts to manage control flow between itself and the user. We evaluate our OS-inspired design in two domains where the limited context windows of modern LLMs severely handicaps their performance: document analysis, where MemGPT is able to analyze large documents that far exceed the underlying LLM's context window, and multi-session chat, where MemGPT can create conversational agents that remember, reflect, and evolve dynamically through long-term interactions with their users. We release MemGPT code and data for our experiments at https://memgpt.ai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.02705v3">Certifying LLM Safety against Adversarial Prompting</a></div>
    <div class="paper-meta">
      📅 2024-02-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07442v1">Game Agent Driven by Free-Form Text Command: Using LLM-based Code Generation and Behavior Branch</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 This paper is posted at JSAI 2024 Conference
    </div>
    <details class="paper-abstract">
      Several attempts have been made to implement text command control for game agents. However, current technologies are limited to processing predefined format commands. This paper proposes a pioneering text command control system for a game agent that can understand natural language commands expressed in free-form. The proposed system uses a large language model (LLM) for code generation to interpret and transform natural language commands into behavior branch, a proposed knowledge expression based on behavior trees, which facilitates execution by the game agent. This study conducted empirical validation within a game environment that simulates a Pok\'emon game and involved multiple participants. The results confirmed the system's ability to understand and carry out natural language commands, representing a noteworthy in the realm of real-time language interactive game agents. Notice for the use of this material. The copyright of this material is retained by the Japanese Society for Artificial Intelligence (JSAI). This material is published here with the agreement of JSAI. Please be complied with Copyright Law of Japan if any users wish to reproduce, make derivative work, distribute or make available to the public any part or whole thereof. All Rights Reserved, Copyright (C) The Japanese Society for Artificial Intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07405v1">Dólares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between Spanish and English</a></div>
    <div class="paper-meta">
      📅 2024-02-12
      | 💬 10 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Despite Spanish's pivotal role in the global finance industry, a pronounced gap exists in Spanish financial natural language processing (NLP) and application studies compared to English, especially in the era of large language models (LLMs). To bridge this gap, we unveil Tois\'on de Oro, the first bilingual framework that establishes instruction datasets, finetuned LLMs, and evaluation benchmark for financial LLMs in Spanish joint with English. We construct a rigorously curated bilingual instruction dataset including over 144K Spanish and English samples from 15 datasets covering 7 tasks. Harnessing this, we introduce FinMA-ES, an LLM designed for bilingual financial applications. We evaluate our model and existing LLMs using FLARE-ES, the first comprehensive bilingual evaluation benchmark with 21 datasets covering 9 tasks. The FLARE-ES benchmark results reveal a significant multilingual performance gap and bias in existing LLMs. FinMA-ES models surpass SOTA LLMs such as GPT-4 in Spanish financial tasks, due to strategic instruction tuning and leveraging data from diverse linguistic resources, highlighting the positive impact of cross-linguistic transfer. All our datasets, models, and benchmarks have been released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07401v1">Can LLMs Produce Faithful Explanations For Fact-checking? Towards Faithful Explainable Fact-Checking via Multi-Agent Debate</a></div>
    <div class="paper-meta">
      📅 2024-02-12
    </div>
    <details class="paper-abstract">
      Fact-checking research has extensively explored verification but less so the generation of natural-language explanations, crucial for user trust. While Large Language Models (LLMs) excel in text generation, their capability for producing faithful explanations in fact-checking remains underexamined. Our study investigates LLMs' ability to generate such explanations, finding that zero-shot prompts often result in unfaithfulness. To address these challenges, we propose the Multi-Agent Debate Refinement (MADR) framework, leveraging multiple LLMs as agents with diverse roles in an iterative refining process aimed at enhancing faithfulness in generated explanations. MADR ensures that the final explanation undergoes rigorous validation, significantly reducing the likelihood of unfaithful elements and aligning closely with the provided evidence. Experimental results demonstrate that MADR significantly improves the faithfulness of LLM-generated explanations to the evidence, advancing the credibility and trustworthiness of these explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07321v1">Summing Up the Facts: Additive Mechanisms Behind Factual Recall in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-11
      | 💬 NeurIPS 2023 Attributing Model Behaviour at Scale Workshop
    </div>
    <details class="paper-abstract">
      How do transformer-based large language models (LLMs) store and retrieve knowledge? We focus on the most basic form of this task -- factual recall, where the model is tasked with explicitly surfacing stored facts in prompts of form `Fact: The Colosseum is in the country of'. We find that the mechanistic story behind factual recall is more complex than previously thought. It comprises several distinct, independent, and qualitatively different mechanisms that additively combine, constructively interfering on the correct attribute. We term this generic phenomena the additive motif: models compute through summing up multiple independent contributions. Each mechanism's contribution may be insufficient alone, but summing results in constructive interfere on the correct answer. In addition, we extend the method of direct logit attribution to attribute an attention head's output to individual source tokens. We use this technique to unpack what we call `mixed heads' -- which are themselves a pair of two separate additive updates from different source tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05880v2">Generative Echo Chamber? Effects of LLM-Powered Search Systems on Diverse Information Seeking</a></div>
    <div class="paper-meta">
      📅 2024-02-10
      | 💬 Accepted in CHI'24. Supplementary material will be available online with the official submission in CHI 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) powered conversational search systems have already been used by hundreds of millions of people, and are believed to bring many benefits over conventional search. However, while decades of research and public discourse interrogated the risk of search systems in increasing selective exposure and creating echo chambers -- limiting exposure to diverse opinions and leading to opinion polarization, little is known about such a risk of LLM-powered conversational search. We conduct two experiments to investigate: 1) whether and how LLM-powered conversational search increases selective exposure compared to conventional search; 2) whether and how LLMs with opinion biases that either reinforce or challenge the user's view change the effect. Overall, we found that participants engaged in more biased information querying with LLM-powered conversational search, and an opinionated LLM reinforcing their views exacerbated this bias. These results present critical implications for the development of LLMs and conversational search systems, and the policy governing these technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05406v2">Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes</a></div>
    <div class="paper-meta">
      📅 2024-02-09
      | 💬 15 pages, 4 fiigures, 15 tables
    </div>
    <details class="paper-abstract">
      Given the generational gap in available hardware between lay practitioners and the most endowed institutions, LLMs are becoming increasingly inaccessible as they grow in size. Whilst many approaches have been proposed to compress LLMs to make their resource consumption manageable, these methods themselves tend to be resource intensive, putting them out of the reach of the very user groups they target. In this work, we explore the problem of structured pruning of LLMs using only forward passes. We seek to empower practitioners to prune models so large that their available hardware has just enough memory to run inference. We develop Bonsai, a gradient-free, perturbative pruning method capable of delivering small, fast, and accurate pruned models. We observe that Bonsai outputs pruned models that (i) outperform those generated by more expensive gradient-based structured pruning methods, and (ii) are twice as fast (with comparable accuracy) as those generated by semi-structured pruning methods requiring comparable resources as Bonsai. We also leverage Bonsai to produce a new sub-2B model using a single A6000 that yields state-of-the-art performance on 4/6 tasks on the Huggingface Open LLM leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.06334v1">ExaRanker-Open: Synthetic Explanation for IR using Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-09
    </div>
    <details class="paper-abstract">
      ExaRanker recently introduced an approach to training information retrieval (IR) models, incorporating natural language explanations as additional labels. The method addresses the challenge of limited labeled examples, leading to improvements in the effectiveness of IR models. However, the initial results were based on proprietary language models such as GPT-3.5, which posed constraints on dataset size due to its cost and data privacy. In this paper, we introduce ExaRanker-Open, where we adapt and explore the use of open-source language models to generate explanations. The method has been tested using different LLMs and datasets sizes to better comprehend the effective contribution of data augmentation. Our findings reveal that incorporating explanations consistently enhances neural rankers, with benefits escalating as the LLM size increases. Notably, the data augmentation method proves advantageous even with large datasets, as evidenced by ExaRanker surpassing the target baseline by 0.6 nDCG@10 points in our study. To encourage further advancements by the research community, we have open-sourced both the code and datasets at https://github.com/unicamp-dl/ExaRanker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07940v1">LLMs Among Us: Generative AI Participating in Digital Discourse</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) has great potential to reshape the landscape of many social media platforms. While this can bring promising opportunities, it also raises many threats, such as biases and privacy concerns, and may contribute to the spread of propaganda by malicious actors. We developed the "LLMs Among Us" experimental framework on top of the Mastodon social media platform for bot and human participants to communicate without knowing the ratio or nature of bot and human participants. We built 10 personas with three different LLMs, GPT-4, LLama 2 Chat, and Claude. We conducted three rounds of the experiment and surveyed participants after each round to measure the ability of LLMs to pose as human participants without human detection. We found that participants correctly identified the nature of other users in the experiment only 42% of the time despite knowing the presence of both bots and humans. We also found that the choice of persona had substantially more impact on human perception than the choice of mainstream LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05904v1">FACT-GPT: Fact-Checking Augmentation via Claim Matching with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      Our society is facing rampant misinformation harming public health and trust. To address the societal challenge, we introduce FACT-GPT, a system leveraging Large Language Models (LLMs) to automate the claim matching stage of fact-checking. FACT-GPT, trained on a synthetic dataset, identifies social media content that aligns with, contradicts, or is irrelevant to previously debunked claims. Our evaluation shows that our specialized LLMs can match the accuracy of larger models in identifying related claims, closely mirroring human judgment. This research provides an automated solution for efficient claim matching, demonstrates the potential of LLMs in supporting fact-checkers, and offers valuable resources for further research in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05864v1">Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05863v1">How Well Can LLMs Negotiate? NegotiationArena Platform and Analysis</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      Negotiation is the basis of social interactions; humans negotiate everything from the price of cars to how to share common resources. With rapidly growing interest in using large language models (LLMs) to act as agents on behalf of human users, such LLM agents would also need to be able to negotiate. In this paper, we study how well LLMs can negotiate with each other. We develop NegotiationArena: a flexible framework for evaluating and probing the negotiation abilities of LLM agents. We implemented three types of scenarios in NegotiationArena to assess LLM's behaviors in allocating shared resources (ultimatum games), aggregate resources (trading games) and buy/sell goods (price negotiations). Each scenario allows for multiple turns of flexible dialogues between LLM agents to allow for more complex negotiations. Interestingly, LLM agents can significantly boost their negotiation outcomes by employing certain behavioral tactics. For example, by pretending to be desolate and desperate, LLMs can improve their payoffs by 20\% when negotiating against the standard GPT-4. We also quantify irrational negotiation behaviors exhibited by the LLM agents, many of which also appear in humans. Together, \NegotiationArena offers a new environment to investigate LLM interactions, enabling new insights into LLM's theory of mind, irrationality, and reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05862v1">Let Your Graph Do the Talking: Encoding Structured Data for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      How can we best encode structured data into sequential form for use in large language models (LLMs)? In this work, we introduce a parameter-efficient method to explicitly represent structured data for LLMs. Our method, GraphToken, learns an encoding function to extend prompts with explicit structured information. Unlike other work which focuses on limited domains (e.g. knowledge graph representation), our work is the first effort focused on the general encoding of structured data to be used for various reasoning tasks. We show that explicitly representing the graph structure allows significant improvements to graph reasoning tasks. Specifically, we see across the board improvements - up to 73% points - on node, edge and, graph-level tasks from the GraphQA benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05467v1">Rapid Optimization for Jailbreaking LLMs via Subconscious Exploitation and Echopraxia</a></div>
    <div class="paper-meta">
      📅 2024-02-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become prevalent across diverse sectors, transforming human life with their extraordinary reasoning and comprehension abilities. As they find increased use in sensitive tasks, safety concerns have gained widespread attention. Extensive efforts have been dedicated to aligning LLMs with human moral principles to ensure their safe deployment. Despite their potential, recent research indicates aligned LLMs are prone to specialized jailbreaking prompts that bypass safety measures to elicit violent and harmful content. The intrinsic discrete nature and substantial scale of contemporary LLMs pose significant challenges in automatically generating diverse, efficient, and potent jailbreaking prompts, representing a continuous obstacle. In this paper, we introduce RIPPLE (Rapid Optimization via Subconscious Exploitation and Echopraxia), a novel optimization-based method inspired by two psychological concepts: subconsciousness and echopraxia, which describe the processes of the mind that occur without conscious awareness and the involuntary mimicry of actions, respectively. Evaluations across 6 open-source LLMs and 4 commercial LLM APIs show RIPPLE achieves an average Attack Success Rate of 91.5\%, outperforming five current methods by up to 47.0\% with an 8x reduction in overhead. Furthermore, it displays significant transferability and stealth, successfully evading established detection mechanisms. The code of our work is available at \url{https://github.com/SolidShen/RIPPLE_official/tree/official}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.12815v2">Investigating Agency of LLMs in Human-AI Collaboration Tasks</a></div>
    <div class="paper-meta">
      📅 2024-02-08
      | 💬 EACL 2024
    </div>
    <details class="paper-abstract">
      Agency, the capacity to proactively shape events, is central to how humans interact and collaborate. While LLMs are being developed to simulate human behavior and serve as human-like agents, little attention has been given to the Agency that these models should possess in order to proactively manage the direction of interaction and collaboration. In this paper, we investigate Agency as a desirable function of LLMs, and how it can be measured and managed. We build on social-cognitive theory to develop a framework of features through which Agency is expressed in dialogue - indicating what you intend to do (Intentionality), motivating your intentions (Motivation), having self-belief in intentions (Self-Efficacy), and being able to self-adjust (Self-Regulation). We collect a new dataset of 83 human-human collaborative interior design conversations containing 908 conversational snippets annotated for Agency features. Using this dataset, we develop methods for measuring Agency of LLMs. Automatic and human evaluations show that models that manifest features associated with high Intentionality, Motivation, Self-Efficacy, and Self-Regulation are more likely to be perceived as strongly agentive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.02716v1">Understanding the planning of LLM agents: A survey</a></div>
    <div class="paper-meta">
      📅 2024-02-08
      | 💬 9 pages, 2 tables, 2 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) have shown significant intelligence, the progress to leverage LLMs as planning modules of autonomous agents has attracted more attention. This survey provides the first systematic view of LLM-based agents planning, covering recent works aiming to improve planning ability. We provide a taxonomy of existing works on LLM-Agent planning, which can be categorized into Task Decomposition, Plan Selection, External Module, Reflection and Memory. Comprehensive analyses are conducted for each direction, and further challenges for the field of research are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14590v1">Scaling Up LLM Reviews for Google Ads Content Moderation</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful tools for content moderation, but their inference costs and latency make them prohibitive for casual use on large datasets, such as the Google Ads repository. This study proposes a method for scaling up LLM reviews for content moderation in Google Ads. First, we use heuristics to select candidates via filtering and duplicate removal, and create clusters of ads for which we select one representative ad per cluster. We then use LLMs to review only the representative ads. Finally, we propagate the LLM decisions for the representative ads back to their clusters. This method reduces the number of reviews by more than 3 orders of magnitude while achieving a 2x recall compared to a baseline non-LLM model. The success of this approach is a strong function of the representations used in clustering and label propagation; we found that cross-modal similarity representations yield better results than uni-modal representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05318v1">Navigating the Knowledge Sea: Planet-scale answer retrieval using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      Information retrieval is a rapidly evolving field of information retrieval, which is characterized by a continuous refinement of techniques and technologies, from basic hyperlink-based navigation to sophisticated algorithm-driven search engines. This paper aims to provide a comprehensive overview of the evolution of Information Retrieval Technology, with a particular focus on the role of Large Language Models (LLMs) in bridging the gap between traditional search methods and the emerging paradigm of answer retrieval. The integration of LLMs in the realms of response retrieval and indexing signifies a paradigm shift in how users interact with information systems. This paradigm shift is driven by the integration of large language models (LLMs) like GPT-4, which are capable of understanding and generating human-like text, thus enabling them to provide more direct and contextually relevant answers to user queries. Through this exploration, we seek to illuminate the technological milestones that have shaped this journey and the potential future directions in this rapidly changing field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06988v1">Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      To ensure that text generated by large language models (LLMs) is in an expected format, constrained decoding proposes to enforce strict formal language constraints during generation. However, as we show in this work, not only do such methods incur performance overhead during generation, but many of them also significantly impair task accuracy, if they do not correctly align the underlying LLM sub-word vocabularies with external constraints. To address this, we present a novel decoding algorithm, DOMINO, that can enforce constraints in a fully subword-aligned fashion, while leveraging pre-computation and speculative decoding to achieve virtually no overhead and in some cases even almost 2$\times$ speedup over unconstrained decoding -- thereby outperforming existing approaches by a wide margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04636v1">TransLLaMa: LLM-based Simultaneous Translation System</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      Decoder-only large language models (LLMs) have recently demonstrated impressive capabilities in text generation and reasoning. Nonetheless, they have limited applications in simultaneous machine translation (SiMT), currently dominated by encoder-decoder transformers. This study demonstrates that, after fine-tuning on a small dataset comprising causally aligned source and target sentence pairs, a pre-trained open-source LLM can control input segmentation directly by generating a special "wait" token. This obviates the need for a separate policy and enables the LLM to perform English-German and English-Russian SiMT tasks with BLEU scores that are comparable to those of specific state-of-the-art baselines. We also evaluated closed-source models such as GPT-4, which displayed encouraging results in performing the SiMT task without prior training (zero-shot), indicating a promising avenue for enhancing future SiMT systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04609v1">Improving Cross-Domain Low-Resource Text Generation through LLM Post-Editing: A Programmer-Interpreter Approach</a></div>
    <div class="paper-meta">
      📅 2024-02-07
      | 💬 EACL 2024 (findings), short paper, 5 pages
    </div>
    <details class="paper-abstract">
      Post-editing has proven effective in improving the quality of text generated by large language models (LLMs) such as GPT-3.5 or GPT-4, particularly when direct updating of their parameters to enhance text quality is infeasible or expensive. However, relying solely on smaller language models for post-editing can limit the LLMs' ability to generalize across domains. Moreover, the editing strategies in these methods are not optimally designed for text-generation tasks. To address these limitations, we propose a neural programmer-interpreter approach that preserves the domain generalization ability of LLMs when editing their output. The editing actions in this framework are specifically devised for text generation. Extensive experiments demonstrate that the programmer-interpreter significantly enhances GPT-3.5's performance in logical form-to-text conversion and low-resource machine translation, surpassing other state-of-the-art (SOTA) LLM post-editing methods in cross-domain settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14596v1">The Role of LLMs in Sustainable Smart Cities: Applications, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      Smart cities stand as pivotal components in the ongoing pursuit of elevating urban living standards, facilitating the rapid expansion of urban areas while efficiently managing resources through sustainable and scalable innovations. In this regard, as emerging technologies like Artificial Intelligence (AI), the Internet of Things (IoT), big data analytics, and fog and edge computing have become increasingly prevalent, smart city applications grapple with various challenges, including the potential for unauthorized disclosure of confidential and sensitive data. The seamless integration of emerging technologies has played a vital role in sustaining the dynamic pace of their development. This paper explores the substantial potential and applications of Deep Learning (DL), Federated Learning (FL), IoT, Blockchain, Natural Language Processing (NLP), and large language models (LLMs) in optimizing ICT processes within smart cities. We aim to spotlight the vast potential of these technologies as foundational elements that technically strengthen the realization and advancement of smart cities, underscoring their significance in driving innovation within this transformative urban milieu. Our discourse culminates with an exploration of the formidable challenges that DL, FL, IoT, Blockchain, NLP, and LLMs face within these contexts, and we offer insights into potential future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02018v3">The Landscape and Challenges of HPC Research and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-07
    </div>
    <details class="paper-abstract">
      Recently, language models (LMs), especially large language models (LLMs), have revolutionized the field of deep learning. Both encoder-decoder models and prompt-based techniques have shown immense potential for natural language processing and code-based tasks. Over the past several years, many research labs and institutions have invested heavily in high-performance computing, approaching or breaching exascale performance levels. In this paper, we posit that adapting and utilizing such language model-based techniques for tasks in high-performance computing (HPC) would be very beneficial. This study presents our reasoning behind the aforementioned position and highlights how existing ideas can be improved and adapted for HPC tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04380v1">Assured LLM-Based Software Engineering</a></div>
    <div class="paper-meta">
      📅 2024-02-06
      | 💬 6 pages, 1 figure, InteNSE 24: ACM International Workshop on Interpretability, Robustness, and Benchmarking in Neural Software Engineering, April, 2024, Lisbon, Portugal
    </div>
    <details class="paper-abstract">
      In this paper we address the following question: How can we use Large Language Models (LLMs) to improve code independently of a human, while ensuring that the improved code - does not regress the properties of the original code? - improves the original in a verifiable and measurable way? To address this question, we advocate Assured LLM-Based Software Engineering; a generate-and-test approach, inspired by Genetic Improvement. Assured LLMSE applies a series of semantic filters that discard code that fails to meet these twin guarantees. This overcomes the potential problem of LLM's propensity to hallucinate. It allows us to generate code using LLMs, independently of any human. The human plays the role only of final code reviewer, as they would do with code generated by other human engineers. This paper is an outline of the content of the keynote by Mark Harman at the International Workshop on Interpretability, Robustness, and Benchmarking in Neural Software Engineering, Monday 15th April 2024, Lisbon, Portugal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.11123v2">(Why) Is My Prompt Getting Worse? Rethinking Regression Testing for Evolving LLM APIs</a></div>
    <div class="paper-meta">
      📅 2024-02-06
      | 💬 conference version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into software applications. Downstream application developers often access LLMs through APIs provided as a service. However, LLM APIs are often updated silently and scheduled to be deprecated, forcing users to continuously adapt to evolving models. This can cause performance regression and affect prompt design choices, as evidenced by our case study on toxicity detection. Based on our case study, we emphasize the need for and re-examine the concept of regression testing for evolving LLM APIs. We argue that regression testing LLMs requires fundamental changes to traditional testing approaches, due to different correctness notions, prompting brittleness, and non-determinism in LLM APIs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04335v1">LegalLens: Leveraging LLMs for Legal Violation Identification in Unstructured Text</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      In this study, we focus on two main tasks, the first for detecting legal violations within unstructured textual data, and the second for associating these violations with potentially affected individuals. We constructed two datasets using Large Language Models (LLMs) which were subsequently validated by domain expert annotators. Both tasks were designed specifically for the context of class-action cases. The experimental design incorporated fine-tuning models from the BERT family and open-source LLMs, and conducting few-shot experiments using closed-source LLMs. Our results, with an F1-score of 62.69\% (violation identification) and 81.02\% (associating victims), show that our datasets and setups can be used for both tasks. Finally, we publicly release the datasets and the code used for the experiments in order to advance further research in the area of legal natural language processing (NLP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2307.07889v3">LLM Comparative Assessment: Zero-shot NLG Evaluation through Pairwise Comparisons using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-02-06
      | 💬 To Appear at EACL 2024
    </div>
    <details class="paper-abstract">
      Current developments in large language models (LLMs) have enabled impressive zero-shot capabilities across various natural language tasks. An interesting application of these systems is in the automated assessment of natural language generation (NLG), a highly challenging area with great practical benefit. In this paper, we explore two options for exploiting the emergent abilities of LLMs for zero-shot NLG assessment: absolute score prediction, and comparative assessment which uses relative comparisons between pairs of candidates. Though comparative assessment has not been extensively studied in NLG assessment, we note that humans often find it more intuitive to compare two options rather than scoring each one independently. This work examines comparative assessment from multiple perspectives: performance compared to absolute grading; positional biases in the prompt; and efficient ranking in terms of the number of comparisons. We illustrate that LLM comparative assessment is a simple, general and effective approach for NLG assessment. For moderate-sized open-source LLMs, such as FlanT5 and Llama2-chat, comparative assessment is superior to prompt scoring, and in many cases can achieve performance competitive with state-of-the-art methods. Additionally, we demonstrate that LLMs often exhibit strong positional biases when making pairwise comparisons, and we propose debiasing methods that can further improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03804v1">ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      Sparse computation offers a compelling solution for the inference of Large Language Models (LLMs) in low-resource scenarios by dynamically skipping the computation of inactive neurons. While traditional approaches focus on ReLU-based LLMs, leveraging zeros in activation values, we broaden the scope of sparse LLMs beyond zero activation values. We introduce a general method that defines neuron activation through neuron output magnitudes and a tailored magnitude threshold, demonstrating that non-ReLU LLMs also exhibit sparse activation. To find the most efficient activation function for sparse computation, we propose a systematic framework to examine the sparsity of LLMs from three aspects: the trade-off between sparsity and performance, the predictivity of sparsity, and the hardware affinity. We conduct thorough experiments on LLMs utilizing different activation functions, including ReLU, SwiGLU, ReGLU, and ReLU$^2$. The results indicate that models employing ReLU$^2$ excel across all three evaluation aspects, highlighting its potential as an efficient activation function for sparse LLMs. We will release the code to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03720v1">Similarity-based Neighbor Selection for Graph LLMs</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) present unique challenges for direct processing by Language Learning Models (LLMs), yet their extensive commonsense knowledge and robust reasoning capabilities offer great promise for node classification in TAGs. Prior research in this field has grappled with issues such as over-squashing, heterophily, and ineffective graph information integration, further compounded by inconsistencies in dataset partitioning and underutilization of advanced LLMs. To address these challenges, we introduce Similarity-based Neighbor Selection (SNS). Using SimCSE and advanced neighbor selection techniques, SNS effectively improves the quality of selected neighbors, thereby improving graph representation and alleviating issues like over-squashing and heterophily. Besides, as an inductive and training-free approach, SNS demonstrates superior generalization and scalability over traditional GNN methods. Our comprehensive experiments, adhering to standard dataset partitioning practices, demonstrate that SNS, through simple prompt interactions with LLMs, consistently outperforms vanilla GNNs and achieves state-of-the-art results on datasets like PubMed in node classification, showcasing LLMs' potential in graph structure understanding. Our research further underscores the significance of graph structure integration in LLM applications and identifies key factors for their success in node classification. Code is available at https://github.com/ruili33/SNS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.04076v5">Do LLMs exhibit human-like response biases? A case study in survey design</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become more capable, there is growing excitement about the possibility of using LLMs as proxies for humans in real-world tasks where subjective labels are desired, such as in surveys and opinion polling. One widely-cited barrier to the adoption of LLMs as proxies for humans in subjective tasks is their sensitivity to prompt wording - but interestingly, humans also display sensitivities to instruction changes in the form of response biases. We investigate the extent to which LLMs reflect human response biases, if at all. We look to survey design, where human response biases caused by changes in the wordings of "prompts" have been extensively explored in social psychology literature. Drawing from these works, we design a dataset and framework to evaluate whether LLMs exhibit human-like response biases in survey questionnaires. Our comprehensive evaluation of nine models shows that popular open and commercial LLMs generally fail to reflect human-like behavior, particularly in models that have undergone RLHF. Furthermore, even if a model shows a significant change in the same direction as humans, we find that they are sensitive to perturbations that do not elicit significant changes in humans. These results highlight the pitfalls of using LLMs as human proxies, and underscore the need for finer-grained characterizations of model behavior. Our code, dataset, and collected samples are available at https://github.com/lindiatjuatja/BiasMonkey
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03610v1">RAP: Retrieval-Augmented Planning with Contextual Memory for Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      Owing to recent advancements, Large Language Models (LLMs) can now be deployed as agents for increasingly complex decision-making applications in areas including robotics, gaming, and API integration. However, reflecting past experiences in current decision-making processes, an innate human behavior, continues to pose significant challenges. Addressing this, we propose Retrieval-Augmented Planning (RAP) framework, designed to dynamically leverage past experiences corresponding to the current situation and context, thereby enhancing agents' planning capabilities. RAP distinguishes itself by being versatile: it excels in both text-only and multimodal environments, making it suitable for a wide range of tasks. Empirical evaluations demonstrate RAP's effectiveness, where it achieves SOTA performance in textual scenarios and notably enhances multimodal LLM agents' performance for embodied tasks. These results highlight RAP's potential in advancing the functionality and applicability of LLM agents in complex, real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.02643v1">LLM-Enhanced Data Management</a></div>
    <div class="paper-meta">
      📅 2024-02-06
    </div>
    <details class="paper-abstract">
      Machine learning (ML) techniques for optimizing data management problems have been extensively studied and widely deployed in recent five years. However traditional ML methods have limitations on generalizability (adapting to different scenarios) and inference ability (understanding the context). Fortunately, large language models (LLMs) have shown high generalizability and human-competitive abilities in understanding context, which are promising for data management tasks (e.g., database diagnosis, database tuning). However, existing LLMs have several limitations: hallucination, high cost, and low accuracy for complicated tasks. To address these challenges, we design LLMDB, an LLM-enhanced data management paradigm which has generalizability and high inference ability while avoiding hallucination, reducing LLM cost, and achieving high accuracy. LLMDB embeds domain-specific knowledge to avoid hallucination by LLM fine-tuning and prompt engineering. LLMDB reduces the high cost of LLMs by vector databases which provide semantic search and caching abilities. LLMDB improves the task accuracy by LLM agent which provides multiple-round inference and pipeline executions. We showcase three real-world scenarios that LLMDB can well support, including query rewrite, database diagnosis and data analytics. We also summarize the open research challenges of LLMDB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2402.01711v1">LLM on FHIR -- Demystifying Health Records</a></div>
    <div class="paper-meta">
      📅 2024-02-06
      | 💬 Pre-print of the paper submitted to the Call for Papers for the Special Focus Issue on ChatGPT and Large Language Models (LLMs) in Biomedicine and Health at the Journal of the American Medical Informatics Association: https://academic.oup.com/jamia/pages/call-for-papers-for-special-focus-issue
    </div>
    <details class="paper-abstract">
      Objective: To enhance health literacy and accessibility of health information for a diverse patient population by developing a patient-centered artificial intelligence (AI) solution using large language models (LLMs) and Fast Healthcare Interoperability Resources (FHIR) application programming interfaces (APIs). Materials and Methods: The research involved developing LLM on FHIR, an open-source mobile application allowing users to interact with their health records using LLMs. The app is built on Stanford's Spezi ecosystem and uses OpenAI's GPT-4. A pilot study was conducted with the SyntheticMass patient dataset and evaluated by medical experts to assess the app's effectiveness in increasing health literacy. The evaluation focused on the accuracy, relevance, and understandability of the LLM's responses to common patient questions. Results: LLM on FHIR demonstrated varying but generally high degrees of accuracy and relevance in providing understandable health information to patients. The app effectively translated medical data into patient-friendly language and was able to adapt its responses to different patient profiles. However, challenges included variability in LLM responses and the need for precise filtering of health data. Discussion and Conclusion: LLMs offer significant potential in improving health literacy and making health records more accessible. LLM on FHIR, as a pioneering application in this field, demonstrates the feasibility and challenges of integrating LLMs into patient care. While promising, the implementation and pilot also highlight risks such as inconsistent responses and the importance of replicable output. Future directions include better resource identification mechanisms and executing LLMs on-device to enhance privacy and reduce costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03289v1">Make Every Move Count: LLM-based High-Quality RTL Code Generation Using MCTS</a></div>
    <div class="paper-meta">
      📅 2024-02-05
    </div>
    <details class="paper-abstract">
      Existing large language models (LLMs) for register transfer level code generation face challenges like compilation failures and suboptimal power, performance, and area (PPA) efficiency. This is due to the lack of PPA awareness in conventional transformer decoding algorithms. In response, we present an automated transformer decoding algorithm that integrates Monte Carlo tree-search for lookahead, guiding the transformer to produce compilable, functionally correct, and PPA-optimized code. Empirical evaluation with a fine-tuned language model on RTL codesets shows that our proposed technique consistently generates functionally correct code compared to prompting-only methods and effectively addresses the PPA-unawareness drawback of naive large language models. For the largest design generated by the state-of-the-art LLM (16-bit adder), our technique can achieve a 31.8% improvement in the area-delay product.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03407v1">Enhancing the Stability of LLM-based Speech Generation Systems through Self-Supervised Representations</a></div>
    <div class="paper-meta">
      📅 2024-02-05
      | 💬 10 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are one of the most promising technologies for the next era of speech generation systems, due to their scalability and in-context learning capabilities. Nevertheless, they suffer from multiple stability issues at inference time, such as hallucinations, content skipping or speech repetitions. In this work, we introduce a new self-supervised Voice Conversion (VC) architecture which can be used to learn to encode transitory features, such as content, separately from stationary ones, such as speaker ID or recording conditions, creating speaker-disentangled representations. Using speaker-disentangled codes to train LLMs for text-to-speech (TTS) allows the LLM to generate the content and the style of the speech only from the text, similarly to humans, while the speaker identity is provided by the decoder of the VC model. Results show that LLMs trained over speaker-disentangled self-supervised representations provide an improvement of 4.7pp in speaker similarity over SOTA entangled representations, and a word error rate (WER) 5.4pp lower. Furthermore, they achieve higher naturalness than human recordings of the LibriTTS test-other dataset. Finally, we show that using explicit reference embedding negatively impacts intelligibility (stability), with WER increasing by 14pp compared to the model that only uses text to infer the style.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.02896v1">LLM Agents in Interaction: Measuring Personality Consistency and Linguistic Alignment in Interacting Populations of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-02-05
      | 💬 To appear in Proceedings of the 1st Personalization of Generative AI Workshop, EACL 2024
    </div>
    <details class="paper-abstract">
      While both agent interaction and personalisation are vibrant topics in research on large language models (LLMs), there has been limited focus on the effect of language interaction on the behaviour of persona-conditioned LLM agents. Such an endeavour is important to ensure that agents remain consistent to their assigned traits yet are able to engage in open, naturalistic dialogues. In our experiments, we condition GPT-3.5 on personality profiles through prompting and create a two-group population of LLM agents using a simple variability-inducing sampling algorithm. We then administer personality tests and submit the agents to a collaborative writing task, finding that different profiles exhibit different degrees of personality consistency and linguistic alignment to their conversational partners. Our study seeks to lay the groundwork for better understanding of dialogue-based interaction between LLMs and highlights the need for new approaches to crafting robust, more human-like LLM personas for interactive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.17072v2">SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity</a></div>
    <div class="paper-meta">
      📅 2024-02-05
    </div>
    <details class="paper-abstract">
      Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable advancements in their ability to generate fitting responses to natural language instructions. However, many current works rely on manual evaluation to judge the quality of generated responses. Since such manual evaluation is time-consuming, it does not easily scale to the evaluation of multiple models and model variants. In this short paper, we propose a straightforward but remarkably effective evaluation metric called SemScore, in which we directly compare model outputs to gold target responses using semantic textual similarity (STS). We conduct a comparative evaluation of the model outputs of 12 prominent instruction-tuned LLMs using 8 widely-used evaluation metrics for text generation. We find that our proposed SemScore metric outperforms all other, in many cases more complex, evaluation metrics in terms of correlation to human evaluation. These findings indicate the utility of our proposed metric for the evaluation of instruction-tuned LLMs.
    </details>
</div>
