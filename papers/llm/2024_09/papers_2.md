# llm - 2024_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16455v1">MultiTalk: Introspective and Extrospective Dialogue for Human-Environment-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown promising results in task planning due to their strong natural language understanding and reasoning capabilities. However, issues such as hallucinations, ambiguities in human instructions, environmental constraints, and limitations in the executing agent's capabilities often lead to flawed or incomplete plans. This paper proposes MultiTalk, an LLM-based task planning methodology that addresses these issues through a framework of introspective and extrospective dialogue loops. This approach helps ground generated plans in the context of the environment and the agent's capabilities, while also resolving uncertainties and ambiguities in the given task. These loops are enabled by specialized systems designed to extract and predict task-specific states, and flag mismatches or misalignments among the human user, the LLM agent, and the environment. Effective feedback pathways between these systems and the LLM planner foster meaningful dialogue. The efficacy of this methodology is demonstrated through its application to robotic manipulation tasks. Experiments and ablations highlight the robustness and reliability of our method, and comparisons with baselines further illustrate the superiority of MultiTalk in task planning for embodied agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16430v1">A Comprehensive Survey of Bias in LLMs: Current Landscape and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 2 Tables, 1 Figure
    </div>
    <details class="paper-abstract">
      Large Language Models(LLMs) have revolutionized various applications in natural language processing (NLP) by providing unprecedented text generation, translation, and comprehension capabilities. However, their widespread deployment has brought to light significant concerns regarding biases embedded within these models. This paper presents a comprehensive survey of biases in LLMs, aiming to provide an extensive review of the types, sources, impacts, and mitigation strategies related to these biases. We systematically categorize biases into several dimensions. Our survey synthesizes current research findings and discusses the implications of biases in real-world applications. Additionally, we critically assess existing bias mitigation techniques and propose future research directions to enhance fairness and equity in LLMs. This survey serves as a foundational resource for researchers, practitioners, and policymakers concerned with addressing and understanding biases in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16395v1">Design and Evaluation of a CDSS for Drug Allergy Management Using LLMs and Pharmaceutical Data Integration</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Medication errors significantly threaten patient safety, leading to adverse drug events and substantial economic burdens on healthcare systems. Clinical Decision Support Systems (CDSSs) aimed at mitigating these errors often face limitations, including reliance on static databases and rule-based algorithms, which can result in high false alert rates and alert fatigue among clinicians. This paper introduces HELIOT, an innovative CDSS for drug allergy management, integrating Large Language Models (LLMs) with a comprehensive pharmaceutical data repository. HELIOT leverages advanced natural language processing capabilities to interpret complex medical texts and synthesize unstructured data, overcoming the limitations of traditional CDSSs. An empirical evaluation using a synthetic patient dataset and expert-verified ground truth demonstrates HELIOT's high accuracy, precision, recall, and F1 score, uniformly reaching 100\% across multiple experimental runs. The results underscore HELIOT's potential to enhance decision support in clinical settings, offering a scalable, efficient, and reliable solution for managing drug allergies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14513v2">Order of Magnitude Speedups for LLM Membership Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model's training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly shadow models, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model's training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art shadow model approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16266v1">REBEL: Rule-based and Experience-enhanced Learning with LLMs for Initial Task Allocation in Multi-Human Multi-Robot Teams</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Multi-human multi-robot teams combine the complementary strengths of humans and robots to tackle complex tasks across diverse applications. However, the inherent heterogeneity of these teams presents significant challenges in initial task allocation (ITA), which involves assigning the most suitable tasks to each team member based on their individual capabilities before task execution. While current learning-based methods have shown promising results, they are often computationally expensive to train, and lack the flexibility to incorporate user preferences in multi-objective optimization and adapt to last-minute changes in real-world dynamic environments. To address these issues, we propose REBEL, an LLM-based ITA framework that integrates rule-based and experience-enhanced learning. By leveraging Retrieval-Augmented Generation, REBEL dynamically retrieves relevant rules and past experiences, enhancing reasoning efficiency. Additionally, REBEL can complement pre-trained RL-based ITA policies, improving situational awareness and overall team performance. Extensive experiments validate the effectiveness of our approach across various settings. More details are available at https://sites.google.com/view/ita-rebel .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07638v2">Can We Count on LLMs? The Fixed-Effect Fallacy and Claims of GPT-4 Capabilities</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      In this paper we explore evaluation of LLM capabilities. We present measurements of GPT-4 performance on several deterministic tasks; each task involves a basic calculation and takes as input parameter some element drawn from a large well-defined population (e.g., count elements in a list, multiply two k-digit numbers, etc). We examine several conditions per-task and perform enough trials so that statistically significant differences can be detected. This allows us to investigate the sensitivity of task-accuracy both to query phrasing and input parameter population. We find that seemingly trivial modifications in the task-prompt or input population can yield differences far larger than can be explained by sampling effects. For example, performance on a simple list-counting task varies with query-phrasing and list-length, but also with list composition (i.e., the thing-to-be-counted) and object frequency (e.g., success when an element accounts for $\approx$ 50\% of a list is different from when it accounts for $\approx$ 70\% etc). We conclude that efforts to quantify LLM capabilities easily succumb to the language-as-fixed-effect fallacy, where experimental observations are improperly generalized beyond what the data supports. A consequence appears to be that intuitions that have been formed based on interactions with humans form a very unreliable guide as to which input modifications should ``make no difference'' to LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01837v1">Code-Survey: An LLM-Driven Methodology for Analyzing Large-Scale Codebases</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Modern software systems like the Linux kernel are among the world's largest and most intricate codebases, continually evolving with new features and increasing complexity. Understanding these systems poses significant challenges due to their scale and the unstructured nature of development artifacts such as commits and mailing list discussions. We introduce Code-Survey, the first LLM-driven methodology designed to systematically explore and analyze large-scale codebases. The central principle behind Code-Survey is to treat LLMs as human participants, acknowledging that software development is also a social activity and thereby enabling the application of established social science techniques. By carefully designing surveys, Code-Survey transforms unstructured data, such as commits, emails, into organized, structured, and analyzable datasets. This enables quantitative analysis of complex software evolution and uncovers valuable insights related to design, implementation, maintenance, reliability, and security. To demonstrate the effectiveness of Code-Survey, we apply it to the Linux kernel's eBPF subsystem. We construct the Linux-bpf dataset, comprising over 670 features and 16,000 commits from the Linux community. Our quantitative analysis uncovers important insights into the evolution of eBPF, such as development patterns, feature interdependencies, and areas requiring attention for reliability and security. The insights have been initially validated by eBPF experts. Furthermore, Code-Survey can be directly applied to other subsystems within Linux and to other large-scale software projects. By providing a versatile tool for systematic analysis, Code-Survey facilitates a deeper understanding of complex software systems, enabling improvements across a variety of domains and supporting a wide range of empirical studies. The code and dataset is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16241v1">LLM Echo Chamber: personalized and automated disinformation</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Recent advancements have showcased the capabilities of Large Language Models like GPT4 and Llama2 in tasks such as summarization, translation, and content review. However, their widespread use raises concerns, particularly around the potential for LLMs to spread persuasive, humanlike misinformation at scale, which could significantly influence public opinion. This study examines these risks, focusing on LLMs ability to propagate misinformation as factual. To investigate this, we built the LLM Echo Chamber, a controlled digital environment simulating social media chatrooms, where misinformation often spreads. Echo chambers, where individuals only interact with like minded people, further entrench beliefs. By studying malicious bots spreading misinformation in this environment, we can better understand this phenomenon. We reviewed current LLMs, explored misinformation risks, and applied sota finetuning techniques. Using Microsoft phi2 model, finetuned with our custom dataset, we generated harmful content to create the Echo Chamber. This setup, evaluated by GPT4 for persuasiveness and harmfulness, sheds light on the ethical concerns surrounding LLMs and emphasizes the need for stronger safeguards against misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03703v1">Human Creativity in the Age of LLMs: Randomized Experiments on Divergent and Convergent Thinking</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Working paper
    </div>
    <details class="paper-abstract">
      Large language models are transforming the creative process by offering unprecedented capabilities to algorithmically generate ideas. While these tools can enhance human creativity when people co-create with them, it's unclear how this will impact unassisted human creativity. We conducted two large pre-registered parallel experiments involving 1,100 participants attempting tasks targeting the two core components of creativity, divergent and convergent thinking. We compare the effects of two forms of large language model (LLM) assistance -- a standard LLM providing direct answers and a coach-like LLM offering guidance -- with a control group receiving no AI assistance, and focus particularly on how all groups perform in a final, unassisted stage. Our findings reveal that while LLM assistance can provide short-term boosts in creativity during assisted tasks, it may inadvertently hinder independent creative performance when users work without assistance, raising concerns about the long-term impact on human creativity and cognition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2409.13884v1">A Multi-LLM Debiasing Framework</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful tools with the potential to benefit society immensely, yet, they have demonstrated biases that perpetuate societal inequalities. Despite significant advancements in bias mitigation techniques using data augmentation, zero-shot prompting, and model fine-tuning, biases continuously persist, including subtle biases that may elude human detection. Recent research has shown a growing interest in multi-LLM approaches, which have been demonstrated to be effective in improving the quality of reasoning and factuality in LLMs. Building on this approach, we propose a novel multi-LLM debiasing framework aimed at reducing bias in LLMs. Our work is the first to introduce and evaluate two distinct approaches within this framework for debiasing LLMs: a centralized method, where the conversation is facilitated by a single central LLM, and a decentralized method, where all models communicate directly. Our findings reveal that our multi-LLM framework significantly reduces bias in LLMs, outperforming the baseline method across several social groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16221v2">Do LLMs Know When to NOT Answer? Investigating Abstention Abilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 pages (excluding limitations, references and appendix) and 5 figures
    </div>
    <details class="paper-abstract">
      Abstention Ability (AA) is a critical aspect of Large Language Model (LLM) reliability, referring to an LLM's capability to withhold responses when uncertain or lacking a definitive answer, without compromising performance. Although previous studies have attempted to improve AA, they lack a standardised evaluation method and remain unsuitable for black-box models where token prediction probabilities are inaccessible. This makes comparative analysis challenging, especially for state-of-the-art closed-source commercial LLMs. This paper bridges this gap by introducing a black-box evaluation approach and a new dataset, Abstain-QA, crafted to rigorously assess AA across varied question types (answerable and unanswerable), domains (well-represented and under-represented), and task types (fact centric and reasoning). We also propose a new confusion matrix, the ''Answerable-Unanswerable Confusion Matrix'' (AUCM) which serves as the basis for evaluating AA, by offering a structured and precise approach for assessment. Finally, we explore the impact of three prompting strategies-Strict Prompting, Verbal Confidence Thresholding, and Chain-of-Thought (CoT)-on improving AA. Our results indicate that even powerful models like GPT-4, Mixtral 8x22b encounter difficulties with abstention; however, strategic approaches such as Strict prompting and CoT can enhance this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18807v1">LLM With Tools: A Survey</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The integration of tools in augmenting large language models presents a novel approach toward enhancing the efficiency and accuracy of these models in handling specific, complex tasks. This paper delves into the methodology,challenges, and developments in the realm of teaching LLMs to use external tools, thereby pushing the boundaries of their capabilities beyond pre-existing knowledge bases. We introduce a standardized paradigm for tool integration guided by a series of functions that map user instructions to actionable plans and their execution, emphasizing the significance of understanding user intent, tool selection, and dynamic plan adjustment. Our exploration reveals the various challenges encountered, such as tool invocation timing, selection accuracy, and the need for robust reasoning processes. In addressing these challenges, we investigate techniques within the context of fine-tuning and incontext learning paradigms, highlighting innovative approaches to ensure diversity, augment datasets, and improve generalization.Furthermore, we investigate a perspective on enabling LLMs to not only utilize but also autonomously create tools, which may redefine their role from mere tool users to tool creators. Finally,we reproduced Chameleon's results on ScienceQA and analyzed the code structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01833v2">Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as jailbreaks, seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a simple multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pro, Gemini-Ultra, LlaMA-2 70b and LlaMA-3 70b Chat, and Anthropic Chat. Our results demonstrate the strong efficacy of Crescendo, with it achieving high attack success rates across all evaluated models and tasks. Furthermore, we present Crescendomation, a tool that automates the Crescendo attack and demonstrate its efficacy against state-of-the-art models through our evaluations. Crescendomation surpasses other state-of-the-art jailbreaking techniques on the AdvBench subset dataset, achieving 29-61% higher performance on GPT-4 and 49-71% on Gemini-Pro. Finally, we also demonstrate Crescendo's ability to jailbreak multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.03175v2">Beyond the Black Box: A Statistical Model for LLM Reasoning and Inference</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Bayesian learning model to explain the behavior of Large Language Models (LLMs), focusing on their core optimization metric of next token prediction. We develop a theoretical framework based on an ideal generative text model represented by a multinomial transition probability matrix with a prior, and examine how LLMs approximate this matrix. Key contributions include: (i) a continuity theorem relating embeddings to multinomial distributions, (ii) a demonstration that LLM text generation aligns with Bayesian learning principles, (iii) an explanation for the emergence of in-context learning in larger models, (iv) empirical validation using visualizations of next token probabilities from an instrumented Llama model Our findings provide new insights into LLM functioning, offering a statistical foundation for understanding their capabilities and limitations. This framework has implications for LLM design, training, and application, potentially guiding future developments in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15491v2">Open Conversational LLMs do not know most Spanish words</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Procesamiento del Lenguaje Natural, 73, 95-108
    </div>
    <details class="paper-abstract">
      The growing interest in Large Language Models (LLMs) and in particular in conversational models with which users can interact has led to the development of a large number of open-source chat LLMs. These models are evaluated on a wide range of benchmarks to assess their capabilities in answering questions or solving problems on almost any possible topic or to test their ability to reason or interpret texts. Instead, the evaluation of the knowledge that these models have of the languages has received much less attention. For example, the words that they can recognize and use in different languages. In this paper, we evaluate the knowledge that open-source chat LLMs have of Spanish words by testing a sample of words in a reference dictionary. The results show that open-source chat LLMs produce incorrect meanings for an important fraction of the words and are not able to use most of the words correctly to write sentences with context. These results show how Spanish is left behind in the open-source LLM race and highlight the need to push for linguistic fairness in conversational LLMs ensuring that they provide similar performance across languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16005v1">Bridging Speech and Text: Enhancing ASR with Pinyin-to-Character Pre-training in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Accepted by ISCSLP2024-Special session-Speech Processing in LLM Era
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) with pre-trained speech models has opened up new avenues in automatic speech recognition (ASR). While LLMs excel in multimodal understanding tasks, effectively leveraging their capabilities for ASR remains a significant challenge. This paper presents a novel training approach to enhance LLM performance in ASR tasks. We propose pre-training LLMs on Pinyin embedding sequences, which represent pronunciation features, to generate corresponding Chinese characters. This step enables the LLM to adapt to generating text from pronunciation features before encountering real speech data. Furthermore, we fine-tune the LoRA parameters to enhance the LLM's understanding of speech modality information. In AISHELL-1 corpus, our approach yields a 9.5% relative improvement in ASR tasks compared to the baseline without Pinyi-to-Character pre-training. Additionally, incorporating auxiliary text data for Pinyi-to-Character pre-training further boosts performance, achieving a 19.0% relative improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15979v1">Finetuning LLMs for Comparative Assessment Tasks</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automated assessment in natural language generation is a challenging task. Instruction-tuned large language models (LLMs) have shown promise in reference-free evaluation, particularly through comparative assessment. However, the quadratic computational complexity of pairwise comparisons limits its scalability. To address this, efficient comparative assessment has been explored by applying comparative strategies on zero-shot LLM probabilities. We propose a framework for finetuning LLMs for comparative assessment to align the model's output with the target distribution of comparative probabilities. By training on soft probabilities, our approach improves state-of-the-art performance while maintaining high performance with an efficient subset of comparisons.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10553v2">"Flipped" University: LLM-Assisted Lifelong Learning Environment</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 Pre-print version, accepted for 31st International Conference on Neural Information Processing (ICONIP2024), 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The rapid development of artificial intelligence technologies, particularly Large Language Models (LLMs), has revolutionized the landscape of lifelong learning. This paper introduces a conceptual framework for a self-constructed lifelong learning environment supported by LLMs. It highlights the inadequacies of traditional education systems in keeping pace with the rapid deactualization of knowledge and skills. The proposed framework emphasizes the transformation from institutionalized education to personalized, self-driven learning. It leverages the natural language capabilities of LLMs to provide dynamic and adaptive learning experiences, facilitating the creation of personal intellectual agents that assist in knowledge acquisition. The framework integrates principles of lifelong learning, including the necessity of building personal world models, the dual modes of learning (training and exploration), and the creation of reusable learning artifacts. Additionally, it underscores the importance of curiosity-driven learning and reflective practices in maintaining an effective learning trajectory. The paper envisions the evolution of educational institutions into "flipped" universities, focusing on supporting global knowledge consistency rather than merely structuring and transmitting knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15915v1">Planning in the Dark: LLM-Symbolic Planning Pipeline without Experts</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 8 main body pages, 10 appendix pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in solving natural language-described planning tasks, but their direct use often leads to inconsistent reasoning and hallucination. While hybrid LLM-symbolic planning pipelines have emerged as a more robust alternative, they typically require extensive expert intervention to refine and validate generated action schemas. It not only limits scalability but also introduces a potential for biased interpretation, as a single expert's interpretation of ambiguous natural language descriptions might not align with the user's actual intent. To address this, we propose a novel approach that constructs an action schema library to generate multiple candidates, accounting for the diverse possible interpretations of natural language descriptions. We further introduce a semantic validation and ranking module that automatically filter and rank the generated schemas and plans without expert-in-the-loop. The experiments showed our pipeline maintains superiority in planning over the direct LLM planning approach. These findings demonstrate the feasibility of a fully automated end-to-end LLM-symbolic planner that requires no expert intervention, opening up the possibility for a broader audience to engage with AI planning with less prerequisite of domain expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15890v1">HLB: Benchmarking LLMs' Humanlikeness in Language Use</a></div>
    <div class="paper-meta">
      📅 2024-09-24
    </div>
    <details class="paper-abstract">
      As synthetic data becomes increasingly prevalent in training language models, particularly through generated dialogue, concerns have emerged that these models may deviate from authentic human language patterns, potentially losing the richness and creativity inherent in human communication. This highlights the critical need to assess the humanlikeness of language models in real-world language use. In this paper, we present a comprehensive humanlikeness benchmark (HLB) evaluating 20 large language models (LLMs) using 10 psycholinguistic experiments designed to probe core linguistic aspects, including sound, word, syntax, semantics, and discourse (see https://huggingface.co/spaces/XufengDuan/HumanLikeness). To anchor these comparisons, we collected responses from over 2,000 human participants and compared them to outputs from the LLMs in these experiments. For rigorous evaluation, we developed a coding algorithm that accurately identified language use patterns, enabling the extraction of response distributions for each task. By comparing the response distributions between human participants and LLMs, we quantified humanlikeness through distributional similarity. Our results reveal fine-grained differences in how well LLMs replicate human responses across various linguistic levels. Importantly, we found that improvements in other performance metrics did not necessarily lead to greater humanlikeness, and in some cases, even resulted in a decline. By introducing psycholinguistic methods to model evaluation, this benchmark offers the first framework for systematically assessing the humanlikeness of LLMs in language use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.15654v1">Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-24
      | 💬 15 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Deploying advanced large language models on edge devices, such as smartphones and robotics, is a growing trend that enhances user data privacy and network connectivity resilience while preserving intelligent capabilities. However, such a task exhibits single-batch computing with incredibly low arithmetic intensity, which poses the significant challenges of huge memory footprint and bandwidth demands on limited edge resources. To address these issues, we introduce Cambricon-LLM, a chiplet-based hybrid architecture with NPU and a dedicated NAND flash chip to enable efficient on-device inference of 70B LLMs. Such a hybrid architecture utilizes both the high computing capability of NPU and the data capacity of the NAND flash chip, with the proposed hardware-tiling strategy that minimizes the data movement overhead between NPU and NAND flash chip. Specifically, the NAND flash chip, enhanced by our innovative in-flash computing and on-die ECC techniques, excels at performing precise lightweight on-die processing. Simultaneously, the NPU collaborates with the flash chip for matrix operations and handles special function computations beyond the flash's on-die processing capabilities. Overall, Cambricon-LLM enables the on-device inference of 70B LLMs at a speed of 3.44 token/s, and 7B LLMs at a speed of 36.34 token/s, which is over 22X to 45X faster than existing flash-offloading technologies, showing the potentiality of deploying powerful LLMs in edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15623v1">Safe Guard: an LLM-agent for Real-time Voice-based Hate Speech Detection in Social Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      In this paper, we present Safe Guard, an LLM-agent for the detection of hate speech in voice-based interactions in social VR (VRChat). Our system leverages Open AI GPT and audio feature extraction for real-time voice interactions. We contribute a system design and evaluation of the system that demonstrates the capability of our approach in detecting hate speech, and reducing false positives compared to currently available approaches. Our results indicate the potential of LLM-based agents in creating safer virtual environments and set the groundwork for further advancements in LLM-driven moderation approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15604v1">Persona-L has Entered the Chat: Leveraging LLM and Ability-based Framework for Personas of People with Complex Needs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      We present Persona-L, a novel approach for creating personas using Large Language Models (LLMs) and an ability-based framework, specifically designed to improve the representation of users with complex needs. Traditional methods of persona creation often fall short of accurately depicting the dynamic and diverse nature of complex needs, resulting in oversimplified or stereotypical profiles. Persona-L enables users to create and interact with personas through a chat interface. Persona-L was evaluated through interviews with UX designers (N=6), where we examined its effectiveness in reflecting the complexities of lived experiences of people with complex needs. We report our findings that indicate the potential of Persona-L to increase empathy and understanding of complex needs while also revealing the need for transparency of data used in persona creation, the role of the language and tone, and the need to provide a more balanced presentation of abilities with constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15594v1">Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 EMNLP Main 2024
    </div>
    <details class="paper-abstract">
      Despite broad interest in modeling spoken dialogue agents, most approaches are inherently "half-duplex" -- restricted to turn-based interaction with responses requiring explicit prompting by the user or implicit tracking of interruption or silence events. Human dialogue, by contrast, is "full-duplex" allowing for rich synchronicity in the form of quick and dynamic turn-taking, overlapping speech, and backchanneling. Technically, the challenge of achieving full-duplex dialogue with LLMs lies in modeling synchrony as pre-trained LLMs do not have a sense of "time". To bridge this gap, we propose Synchronous LLMs for full-duplex spoken dialogue modeling. We design a novel mechanism to integrate time information into Llama3-8b so that they run synchronously with the real-world clock. We also introduce a training recipe that uses 212k hours of synthetic spoken dialogue data generated from text dialogue data to create a model that generates meaningful and natural spoken dialogue, with just 2k hours of real-world spoken dialogue data. Synchronous LLMs outperform state-of-the-art in dialogue meaningfulness while maintaining naturalness. Finally, we demonstrate the model's ability to participate in full-duplex dialogue by simulating interaction between two agents trained on different datasets, while considering Internet-scale latencies of up to 240 ms. Webpage: https://syncllm.cs.washington.edu/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13276v2">When LLMs Meets Acoustic Landmarks: An Efficient Approach to Integrate Speech into Large Language Models for Depression Detection</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Depression is a critical concern in global mental health, prompting extensive research into AI-based detection methods. Among various AI technologies, Large Language Models (LLMs) stand out for their versatility in mental healthcare applications. However, their primary limitation arises from their exclusive dependence on textual input, which constrains their overall capabilities. Furthermore, the utilization of LLMs in identifying and analyzing depressive states is still relatively untapped. In this paper, we present an innovative approach to integrating acoustic speech information into the LLMs framework for multimodal depression detection. We investigate an efficient method for depression detection by integrating speech signals into LLMs utilizing Acoustic Landmarks. By incorporating acoustic landmarks, which are specific to the pronunciation of spoken words, our method adds critical dimensions to text transcripts. This integration also provides insights into the unique speech patterns of individuals, revealing the potential mental states of individuals. Evaluations of the proposed approach on the DAIC-WOZ dataset reveal state-of-the-art results when compared with existing Audio-Text baselines. In addition, this approach is not only valuable for the detection of depression but also represents a new perspective in enhancing the ability of LLMs to comprehend and process speech signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07192v1">PipeFill: Using GPUs During Bubbles in Pipeline-parallel LLM Training</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Training Deep Neural Networks (DNNs) with billions of parameters generally involves pipeline-parallel (PP) execution. Unfortunately, PP model training can use GPUs inefficiently, especially at large scale, due to idle GPU time caused by pipeline bubbles, which are often 15-30% and can exceed 60% of the training job's GPU allocation. To improve the GPU utilization of PP model training, this paper describes PipeFill, which fills pipeline bubbles with execution of other pending jobs. By leveraging bubble GPU time, PipeFill reduces the GPU utilization sacrifice associated with scaling-up of large-model training. To context-switch between fill jobs and the main training job with minimal overhead to the main job, and maximize fill job efficiency, PipeFill carefully fits fill job work to measured bubble durations and GPU memory availability, introduces explicit pipeline-bubble instructions, and orchestrates placement and execution of fill jobs in pipeline bubbles. Experiments show that PipeFill can increase overall utilization by up to 63% for GPUs used in large-scale LLM training, with <2% slowdown of the training job, and 5-15% even for low-scale LLM training. For large-scale LLM training on 8K GPUs, the 63% increase translates to up to 2.6K additional GPUs worth of work completed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19568v2">Are LLMs Good Annotators for Discourse-level Event Relation Extraction?</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated proficiency in a wide array of natural language processing tasks. However, its effectiveness over discourse-level event relation extraction (ERE) tasks remains unexplored. In this paper, we assess the effectiveness of LLMs in addressing discourse-level ERE tasks characterized by lengthy documents and intricate relations encompassing coreference, temporal, causal, and subevent types. Evaluation is conducted using an commercial model, GPT-3.5, and an open-source model, LLaMA-2. Our study reveals a notable underperformance of LLMs compared to the baseline established through supervised learning. Although Supervised Fine-Tuning (SFT) can improve LLMs performance, it does not scale well compared to the smaller supervised baseline model. Our quantitative and qualitative analysis shows that LLMs have several weaknesses when applied for extracting event relations, including a tendency to fabricate event mentions, and failures to capture transitivity rules among relations, detect long distance relations, or comprehend contexts with dense event mentions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15551v1">Revise, Reason, and Recognize: LLM-Based Emotion Recognition via Emotion-Specific Prompts and ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Annotating and recognizing speech emotion using prompt engineering has recently emerged with the advancement of Large Language Models (LLMs), yet its efficacy and reliability remain questionable. In this paper, we conduct a systematic study on this topic, beginning with the proposal of novel prompts that incorporate emotion-specific knowledge from acoustics, linguistics, and psychology. Subsequently, we examine the effectiveness of LLM-based prompting on Automatic Speech Recognition (ASR) transcription, contrasting it with ground-truth transcription. Furthermore, we propose a Revise-Reason-Recognize prompting pipeline for robust LLM-based emotion recognition from spoken language with ASR errors. Additionally, experiments on context-aware learning, in-context learning, and instruction tuning are performed to examine the usefulness of LLM training schemes in this direction. Finally, we investigate the sensitivity of LLMs to minor prompt variations. Experimental results demonstrate the efficacy of the emotion-specific prompts, ASR error correction, and LLM training schemes for LLM-based emotion recognition. Our study aims to refine the use of LLMs in emotion recognition and related domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15523v1">SEAL: Suite for Evaluating API-use of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have limitations in handling tasks that require real-time access to external APIs. While several benchmarks like ToolBench and APIGen have been developed to assess LLMs' API-use capabilities, they often suffer from issues such as lack of generalizability, limited multi-step reasoning coverage, and instability due to real-time API fluctuations. In this paper, we introduce SEAL, an end-to-end testbed designed to evaluate LLMs in real-world API usage. SEAL standardizes existing benchmarks, integrates an agent system for testing API retrieval and planning, and addresses the instability of real-time APIs by introducing a GPT-4-powered API simulator with caching for deterministic evaluations. Our testbed provides a comprehensive evaluation pipeline that covers API retrieval, API calls, and final responses, offering a reliable framework for structured performance comparison in diverse real-world scenarios. SEAL is publicly available, with ongoing updates for new benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15471v1">EvAlignUX: Advancing UX Research through LLM-Supported Exploration of Evaluation Metrics</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Evaluating UX in the context of AI's complexity, unpredictability, and generative nature presents unique challenges. HCI scholars lack sufficient tool support to build knowledge around diverse evaluation metrics and develop comprehensive UX evaluation plans. In this paper, we introduce EvAlignUX, an innovative system grounded in scientific literature and powered by large language models (LLMs), designed to help HCI scholars explore evaluation metrics and their relationship to potential research outcomes. A user study involving 19 HCI scholars revealed that EvAlignUX significantly improved the perceived clarity, specificity, feasibility, and overall quality of their evaluation proposals. The use of EvAlignUX enhanced participants' thought processes, resulting in the creation of a Question Bank that can be used to guide UX Evaluation Development. Additionally, the influence of researchers' backgrounds on their perceived inspiration and concerns about over-reliance on AI highlights future research directions for AI's role in fostering critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15436v1">GenAI Advertising: Risks of Personalizing Ads with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Recent advances in large language models have enabled the creation of highly effective chatbots, which may serve as a platform for targeted advertising. This paper investigates the risks of personalizing advertising in chatbots to their users. We developed a chatbot that embeds personalized product advertisements within LLM responses, inspired by similar forays by AI companies. Our benchmarks show that ad injection impacted certain LLM attribute performance, particularly response desirability. We conducted a between-subjects experiment with 179 participants using chabots with no ads, unlabeled targeted ads, and labeled targeted ads. Results revealed that participants struggled to detect chatbot ads and unlabeled advertising chatbot responses were rated higher. Yet, once disclosed, participants found the use of ads embedded in LLM responses to be manipulative, less trustworthy, and intrusive. Participants tried changing their privacy settings via chat interface rather than the disclosure. Our findings highlight ethical issues with integrating advertising into chatbot responses
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15241v1">Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Given the popularity of generative AI, Large Language Models (LLMs) often consume hundreds or thousands of GPUs for parallelizing and accelerating the training process. Communication overhead becomes more pronounced when training LLMs at scale. To eliminate communication overhead in distributed LLM training, we propose Domino, which provides a generic scheme to hide communication behind computation. By breaking data dependency of a single batch training into smaller independent pieces, Domino pipelines these independent pieces training and provides generic strategy of fine-grained communication and computation overlapping. Extensive results show that, comparing with Megatron-LM, Domino achieves up to 1.3x speedup for LLM training on Nvidia DGX-H100 GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13050v2">Human-Centered LLM-Agent User Interface: A Position Paper</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) -in-the-loop applications have been shown to effectively interpret the human user's commands, make plans, and operate external tools/systems accordingly. Still, the operation scope of the LLM agent is limited to passively following the user, requiring the user to frame his/her needs with regard to the underlying tools/systems. We note that the potential of an LLM-Agent User Interface (LAUI) is much greater. A user mostly ignorant to the underlying tools/systems should be able to work with a LAUI to discover an emergent workflow. Contrary to the conventional way of designing an explorable GUI to teach the user a predefined set of ways to use the system, in the ideal LAUI, the LLM agent is initialized to be proficient with the system, proactively studies the user and his/her needs, and proposes new interaction schemes to the user. To illustrate LAUI, we present Flute X GPT, a concrete example using an LLM agent, a prompt manager, and a flute-tutoring multi-modal software-hardware system to facilitate the complex, real-time user experience of learning to play the flute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15135v1">Controllable Traffic Simulation through LLM-Guided Hierarchical Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Evaluating autonomous driving systems in complex and diverse traffic scenarios through controllable simulation is essential to ensure their safety and reliability. However, existing traffic simulation methods face challenges in their controllability. To address this, this paper proposes a novel diffusion-based and LLM-enhanced traffic simulation framework. Our approach incorporates a unique chain-of-thought (CoT) mechanism, which systematically examines the hierarchical structure of traffic elements and guides LLMs to thoroughly analyze traffic scenario descriptions step by step, enhancing their understanding of complex situations. Furthermore, we propose a Frenet-frame-based cost function framework that provides LLMs with geometrically meaningful quantities, improving their grasp of spatial relationships in a scenario and enabling more accurate cost function generation. Experiments on the Waymo Open Motion Dataset (WOMD) demonstrate that our method handles more intricate descriptions, generates a broader range of scenarios in a controllable manner, and outperforms existing diffusion-based methods in terms of efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15133v1">Don't Use LLMs to Make Relevance Judgments</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Making the relevance judgments for a TREC-style test collection can be complex and expensive. A typical TREC track usually involves a team of six contractors working for 2-4 weeks. Those contractors need to be trained and monitored. Software has to be written to support recording relevance judgments correctly and efficiently. The recent advent of large language models that produce astoundingly human-like flowing text output in response to a natural language prompt has inspired IR researchers to wonder how those models might be used in the relevance judgment collection process. At the ACM SIGIR 2024 conference, a workshop ``LLM4Eval'' provided a venue for this work, and featured a data challenge activity where participants reproduced TREC deep learning track judgments, as was done by Thomas et al (arXiv:2408.08896, arXiv:2309.10621). I was asked to give a keynote at the workshop, and this paper presents that keynote in article form. The bottom-line-up-front message is, don't use LLMs to create relevance judgments for TREC-style evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15127v1">Boosting Healthcare LLMs Through Retrieved Context</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 9 pages, 5 figures, 12 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing, and yet, their factual inaccuracies and hallucinations limits their application, particularly in critical domains like healthcare. Context retrieval methods, by introducing relevant information as input, have emerged as a crucial approach for enhancing LLM factuality and reliability. This study explores the boundaries of context retrieval methods within the healthcare domain, optimizing their components and benchmarking their performance against open and closed alternatives. Our findings reveal how open LLMs, when augmented with an optimized retrieval system, can achieve performance comparable to the biggest private solutions on established healthcare benchmarks (multiple-choice question answering). Recognizing the lack of realism of including the possible answers within the question (a setup only found in medical exams), and after assessing a strong LLM performance degradation in the absence of those options, we extend the context retrieval system in that direction. In particular, we propose OpenMedPrompt a pipeline that improves the generation of more reliable open-ended answers, moving this technology closer to practical application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15072v1">Evaluating the Usability of LLMs in Threat Intelligence Enrichment</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to significantly enhance threat intelligence by automating the collection, preprocessing, and analysis of threat data. However, the usability of these tools is critical to ensure their effective adoption by security professionals. Despite the advanced capabilities of LLMs, concerns about their reliability, accuracy, and potential for generating inaccurate information persist. This study conducts a comprehensive usability evaluation of five LLMs ChatGPT, Gemini, Cohere, Copilot, and Meta AI focusing on their user interface design, error handling, learning curve, performance, and integration with existing tools in threat intelligence enrichment. Utilizing a heuristic walkthrough and a user study methodology, we identify key usability issues and offer actionable recommendations for improvement. Our findings aim to bridge the gap between LLM functionality and user experience, thereby promoting more efficient and accurate threat intelligence practices by ensuring these tools are user-friendly and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15052v1">Brotherhood at WMT 2024: Leveraging LLM-Generated Contextual Conversations for Cross-Lingual Image Captioning</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 Accepted at the Ninth Conference on Machine Translation (WMT24), co-located with EMNLP 2024
    </div>
    <details class="paper-abstract">
      In this paper, we describe our system under the team name Brotherhood for the English-to-Lowres Multi-Modal Translation Task. We participate in the multi-modal translation tasks for English-Hindi, English-Hausa, English-Bengali, and English-Malayalam language pairs. We present a method leveraging multi-modal Large Language Models (LLMs), specifically GPT-4o and Claude 3.5 Sonnet, to enhance cross-lingual image captioning without traditional training or fine-tuning. Our approach utilizes instruction-tuned prompting to generate rich, contextual conversations about cropped images, using their English captions as additional context. These synthetic conversations are then translated into the target languages. Finally, we employ a weighted prompting strategy, balancing the original English caption with the translated conversation to generate captions in the target language. This method achieved competitive results, scoring 37.90 BLEU on the English-Hindi Challenge Set and ranking first and second for English-Hausa on the Challenge and Evaluation Leaderboards, respectively. We conduct additional experiments on a subset of 250 images, exploring the trade-offs between BLEU scores and semantic similarity across various weighting schemes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15027v1">Generative LLM Powered Conversational AI Application for Personalized Risk Assessment: A Case Study in COVID-19</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in various natural language tasks and are increasingly being applied in healthcare domains. This work demonstrates a new LLM-powered disease risk assessment approach via streaming human-AI conversation, eliminating the need for programming required by traditional machine learning approaches. In a COVID-19 severity risk assessment case study, we fine-tune pre-trained generative LLMs (e.g., Llama2-7b and Flan-t5-xl) using a few shots of natural language examples, comparing their performance with traditional classifiers (i.e., Logistic Regression, XGBoost, Random Forest) that are trained de novo using tabular data across various experimental settings. We develop a mobile application that uses these fine-tuned LLMs as its generative AI (GenAI) core to facilitate real-time interaction between clinicians and patients, providing no-code risk assessment through conversational interfaces. This integration not only allows for the use of streaming Questions and Answers (QA) as inputs but also offers personalized feature importance analysis derived from the LLM's attention layers, enhancing the interpretability of risk assessments. By achieving high Area Under the Curve (AUC) scores with a limited number of fine-tuning samples, our results demonstrate the potential of generative LLMs to outperform discriminative classification methods in low-data regimes, highlighting their real-world adaptability and effectiveness. This work aims to fill the existing gap in leveraging generative LLMs for interactive no-code risk assessment and to encourage further research in this emerging field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14988v1">Beyond Fine-tuning: Unleashing the Potential of Continuous Pretraining for Clinical LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant potential in transforming clinical applications. In this study, we investigate the efficacy of four techniques in adapting LLMs for clinical use-cases: continuous pretraining, instruct fine-tuning, NEFTune, and prompt engineering. We employ these methods on Mistral 7B and Mixtral 8x7B models, leveraging a large-scale clinical pretraining dataset of 50 billion tokens and an instruct fine-tuning dataset of 500 million tokens. Our evaluation across various clinical tasks reveals the impact of each technique. While continuous pretraining beyond 250 billion tokens yields marginal improvements on its own, it establishes a strong foundation for instruct fine-tuning. Notably, NEFTune, designed primarily to enhance generation quality, surprisingly demonstrates additional gains on our benchmark. Complex prompt engineering methods further enhance performance. These findings show the importance of tailoring fine-tuning strategies and exploring innovative techniques to optimize LLM performance in the clinical domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.00155v3">LLM in the Shell: Generative Honeypots</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 6 pages. 2 figures. 2 tables
    </div>
    <details class="paper-abstract">
      Honeypots are essential tools in cybersecurity for early detection, threat intelligence gathering, and analysis of attacker's behavior. However, most of them lack the required realism to engage and fool human attackers long-term. Being easy to distinguish honeypots strongly hinders their effectiveness. This can happen because they are too deterministic, lack adaptability, or lack deepness. This work introduces shelLM, a dynamic and realistic software honeypot based on Large Language Models that generates Linux-like shell output. We designed and implemented shelLM using cloud-based LLMs. We evaluated if shelLM can generate output as expected from a real Linux shell. The evaluation was done by asking cybersecurity researchers to use the honeypot and give feedback if each answer from the honeypot was the expected one from a Linux shell. Results indicate that shelLM can create credible and dynamic answers capable of addressing the limitations of current honeypots. ShelLM reached a TNR of 0.90, convincing humans it was consistent with a real Linux shell. The source code and prompts for replicating the experiments have been publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14924v1">Retrieval Augmented Generation (RAG) and Beyond: A Comprehensive Survey on How to Make your LLMs use External Data More Wisely</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) augmented with external data have demonstrated remarkable capabilities in completing real-world tasks. Techniques for integrating external data into LLMs, such as Retrieval-Augmented Generation (RAG) and fine-tuning, are gaining increasing attention and widespread application. Nonetheless, the effective deployment of data-augmented LLMs across various specialized fields presents substantial challenges. These challenges encompass a wide range of issues, from retrieving relevant data and accurately interpreting user intent to fully harnessing the reasoning capabilities of LLMs for complex tasks. We believe that there is no one-size-fits-all solution for data-augmented LLM applications. In practice, underperformance often arises from a failure to correctly identify the core focus of a task or because the task inherently requires a blend of multiple capabilities that must be disentangled for better resolution. In this survey, we propose a RAG task categorization method, classifying user queries into four levels based on the type of external data required and primary focus of the task: explicit fact queries, implicit fact queries, interpretable rationale queries, and hidden rationale queries. We define these levels of queries, provide relevant datasets, and summarize the key challenges and most effective techniques for addressing these challenges. Finally, we discuss three main forms of integrating external data into LLMs: context, small model, and fine-tuning, highlighting their respective strengths, limitations, and the types of problems they are suited to solve. This work aims to help readers thoroughly understand and decompose the data requirements and key bottlenecks in building LLM applications, offering solutions to the different challenges and serving as a guide to systematically developing such applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.15200v2">DeMPT: Decoding-enhanced Multi-phase Prompt Tuning for Making LLMs Be Better Context-aware Translators</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 Accepted by EMNLP 2024
    </div>
    <details class="paper-abstract">
      Generally, the decoder-only large language models (LLMs) are adapted to context-aware neural machine translation (NMT) in a concatenating way, where LLMs take the concatenation of the source sentence (i.e., intra-sentence context) and the inter-sentence context as the input, and then to generate the target tokens sequentially. This adaptation strategy, i.e., concatenation mode, considers intra-sentence and inter-sentence contexts with the same priority, despite an apparent difference between the two kinds of contexts. In this paper, we propose an alternative adaptation approach, named Decoding-enhanced Multi-phase Prompt Tuning (DeMPT), to make LLMs discriminately model and utilize the inter- and intra-sentence context and more effectively adapt LLMs to context-aware NMT. First, DeMPT divides the context-aware NMT process into three separate phases. During each phase, different continuous prompts are introduced to make LLMs discriminately model various information. Second, DeMPT employs a heuristic way to further discriminately enhance the utilization of the source-side inter- and intra-sentence information at the final decoding phase. Experiments show that our approach significantly outperforms the concatenation method, and further improves the performance of LLMs in discourse modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14879v1">Privacy Policy Analysis through Prompt Engineering for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Privacy policies are often obfuscated by their complexity, which impedes transparency and informed consent. Conventional machine learning approaches for automatically analyzing these policies demand significant resources and substantial domain-specific training, causing adaptability issues. Moreover, they depend on extensive datasets that may require regular maintenance due to changing privacy concerns. In this paper, we propose, apply, and assess PAPEL (Privacy Policy Analysis through Prompt Engineering for LLMs), a framework harnessing the power of Large Language Models (LLMs) through prompt engineering to automate the analysis of privacy policies. PAPEL aims to streamline the extraction, annotation, and summarization of information from these policies, enhancing their accessibility and comprehensibility without requiring additional model training. By integrating zero-shot, one-shot, and few-shot learning approaches and the chain-of-thought prompting in creating predefined prompts and prompt templates, PAPEL guides LLMs to efficiently dissect, interpret, and synthesize the critical aspects of privacy policies into user-friendly summaries. We demonstrate the effectiveness of PAPEL with two applications: (i) annotation and (ii) contradiction analysis. We assess the ability of several LLaMa and GPT models to identify and articulate data handling practices, offering insights comparable to existing automated analysis approaches while reducing training efforts and increasing the adaptability to new analytical needs. The experiments demonstrate that the LLMs PAPEL utilizes (LLaMA and Chat GPT models) achieve robust performance in privacy policy annotation, with F1 scores reaching 0.8 and above (using the OPP-115 gold standard), underscoring the effectiveness of simpler prompts across various advanced language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14858v1">LLMs' ways of seeing User Personas</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 At IndiaHCI 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), which have gained significant traction in recent years, also function as big structured repositories of data. User personas are a significant and widely utilized method in HCI. This study aims to investigate how LLMs, in their role as data repositories, interpret user personas. Our focus is specifically on personas within the Indian context, seeking to understand how LLMs would interpret such culturally specific personas. To achieve this, we conduct both quantitative and qualitative analyses. This multifaceted approach allows us a primary understanding of the interpretative capabilities of LLMs concerning personas within the Indian context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13447v2">AQA: Adaptive Question Answering in a Society of LLMs via Contextual Multi-Armed Bandit</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      In question answering (QA), different questions can be effectively addressed with different answering strategies. Some require a simple lookup, while others need complex, multi-step reasoning to be answered adequately. This observation motivates the development of a dynamic method that adaptively selects the most suitable QA strategy for each question, enabling more efficient and effective systems capable of addressing a broader range of question types. To this aim, we build on recent advances in the orchestration of multiple large language models (LLMs) and formulate adaptive QA as a dynamic orchestration challenge. We define this as a contextual multi-armed bandit problem, where the context is defined by the characteristics of the incoming question and the action space consists of potential communication graph configurations among the LLM agents. We then train a linear upper confidence bound model to learn an optimal mapping between different question types and their corresponding optimal multi-LLM communication graph representation. Our experiments show that the proposed solution is viable for adaptive orchestration of a QA system with multiple modules, as it combines the superior performance of more complex strategies while avoiding their costs when simpler strategies suffice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14800v1">Choose the Final Translation from NMT and LLM hypotheses Using MBR Decoding: HW-TSC's Submission to the WMT24 General MT Shared Task</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 10 pages, 4 figures, 2 Tables, EMNLP2024
    </div>
    <details class="paper-abstract">
      This paper presents the submission of Huawei Translate Services Center (HW-TSC) to the WMT24 general machine translation (MT) shared task, where we participate in the English to Chinese (en2zh) language pair. Similar to previous years' work, we use training strategies such as regularized dropout, bidirectional training, data diversification, forward translation, back translation, alternated training, curriculum learning, and transductive ensemble learning to train the neural machine translation (NMT) model based on the deep Transformer-big architecture. The difference is that we also use continue pre-training, supervised fine-tuning, and contrastive preference optimization to train the large language model (LLM) based MT model. By using Minimum Bayesian risk (MBR) decoding to select the final translation from multiple hypotheses for NMT and LLM-based MT models, our submission receives competitive results in the final evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.08092v2">A Survey of Resource-efficient LLM and Multimodal Foundation Models</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large foundation models, including large language models (LLMs), vision transformers (ViTs), diffusion, and LLM-based multimodal models, are revolutionizing the entire machine learning lifecycle, from training to deployment. However, the substantial advancements in versatility and performance these models offer come at a significant cost in terms of hardware resources. To support the growth of these large models in a scalable and environmentally sustainable way, there has been a considerable focus on developing resource-efficient strategies. This survey delves into the critical importance of such research, examining both algorithmic and systemic aspects. It offers a comprehensive analysis and valuable insights gleaned from existing literature, encompassing a broad array of topics from cutting-edge model architectures and training/serving algorithms to practical system designs and implementations. The goal of this survey is to provide an overarching understanding of how current approaches are tackling the resource challenges posed by large foundation models and to potentially inspire future breakthroughs in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14739v1">AmpAgent: An LLM-based Multi-Agent System for Multi-stage Amplifier Schematic Design from Literature for Process and Performance Porting</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Multi-stage amplifiers are widely applied in analog circuits. However, their large number of components, complex transfer functions, and intricate pole-zero distributions necessitate extensive manpower for derivation and param sizing to ensure their stability. In order to achieve efficient derivation of the transfer function and simplify the difficulty of circuit design, we propose AmpAgent: a multi-agent system based on large language models (LLMs) for efficiently designing such complex amplifiers from literature with process and performance porting. AmpAgent is composed of three agents: Literature Analysis Agent, Mathematics Reasoning Agent and Device Sizing Agent. They are separately responsible for retrieving key information (e.g. formulas and transfer functions) from the literature, decompose the whole circuit's design problem by deriving the key formulas, and address the decomposed problem iteratively. AmpAgent was employed in the schematic design of seven types of multi-stage amplifiers with different compensation techniques. In terms of design efficiency, AmpAgent has reduced the number of iterations by 1.32$ \sim $4${\times}$ and execution time by 1.19$ \sim $2.99${\times}$ compared to conventional optimization algorithms, with a success rate increased by 1.03$ \sim $6.79${\times}$. In terms of circuit performance, it has improved by 1.63$ \sim $27.25${\times}$ compared to the original literature. The findings suggest that LLMs could play a crucial role in the field of complex analog circuit schematic design, as well as process and performance porting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15395v1">Parse Trees Guided LLM Prompt Compression</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Offering rich contexts to Large Language Models (LLMs) has shown to boost the performance in various tasks, but the resulting longer prompt would increase the computational cost and might exceed the input limit of LLMs. Recently, some prompt compression methods have been suggested to shorten the length of prompts by using language models to generate shorter prompts or by developing computational models to select important parts of original prompt. The generative compression methods would suffer from issues like hallucination, while the selective compression methods have not involved linguistic rules and overlook the global structure of prompt. To this end, we propose a novel selective compression method called PartPrompt. It first obtains a parse tree for each sentence based on linguistic rules, and calculates local information entropy for each node in a parse tree. These local parse trees are then organized into a global tree according to the hierarchical structure such as the dependency of sentences, paragraphs, and sections. After that, the root-ward propagation and leaf-ward propagation are proposed to adjust node values over the global tree. Finally, a recursive algorithm is developed to prune the global tree based on the adjusted node values. The experiments show that PartPrompt receives the state-of-the-art performance across various datasets, metrics, compression ratios, and target LLMs for inference. The in-depth ablation studies confirm the effectiveness of designs in PartPrompt, and other additional experiments also demonstrate its superiority in terms of the coherence of compressed prompts and in the extreme long prompt scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11346v2">Decision support system for Forest fire management using Ontology with Big Data and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Forests are crucial for ecological balance, but wildfires, a major cause of forest loss, pose significant risks. Fire weather indices, which assess wildfire risk and predict resource demands, are vital. With the rise of sensor networks in fields like healthcare and environmental monitoring, semantic sensor networks are increasingly used to gather climatic data such as wind speed, temperature, and humidity. However, processing these data streams to determine fire weather indices presents challenges, underscoring the growing importance of effective forest fire detection. This paper discusses using Apache Spark for early forest fire detection, enhancing fire risk prediction with meteorological and geographical data. Building on our previous development of Semantic Sensor Network (SSN) ontologies and Semantic Web Rules Language (SWRL) for managing forest fires in Monesterial Natural Park, we expanded SWRL to improve a Decision Support System (DSS) using a Large Language Models (LLMs) and Spark framework. We implemented real-time alerts with Spark streaming, tailored to various fire scenarios, and validated our approach using ontology metrics, query-based evaluations, LLMs score precision, F1 score, and recall measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14729v1">PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks. In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts. By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06837v2">How Aligned are Human Chart Takeaways and LLM Predictions? A Case Study on Bar Charts with Varying Layouts</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 IEEE Transactions on Visualization and Computer Graphics (Proc. VIS 2024)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been adopted for a variety of visualizations tasks, but how far are we from perceptually aware LLMs that can predict human takeaways? Graphical perception literature has shown that human chart takeaways are sensitive to visualization design choices, such as spatial layouts. In this work, we examine the extent to which LLMs exhibit such sensitivity when generating takeaways, using bar charts with varying spatial layouts as a case study. We conducted three experiments and tested four common bar chart layouts: vertically juxtaposed, horizontally juxtaposed, overlaid, and stacked. In Experiment 1, we identified the optimal configurations to generate meaningful chart takeaways by testing four LLMs, two temperature settings, nine chart specifications, and two prompting strategies. We found that even state-of-the-art LLMs struggled to generate semantically diverse and factually accurate takeaways. In Experiment 2, we used the optimal configurations to generate 30 chart takeaways each for eight visualizations across four layouts and two datasets in both zero-shot and one-shot settings. Compared to human takeaways, we found that the takeaways LLMs generated often did not match the types of comparisons made by humans. In Experiment 3, we examined the effect of chart context and data on LLM takeaways. We found that LLMs, unlike humans, exhibited variation in takeaway comparison types for different bar charts using the same bar layout. Overall, our case study evaluates the ability of LLMs to emulate human interpretations of data and points to challenges and opportunities in using LLMs to predict human chart takeaways.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.17218v3">Prompting Techniques for Reducing Social Bias in LLMs through System 1 and System 2 Cognitive Processes</a></div>
    <div class="paper-meta">
      📅 2024-09-23
      | 💬 Pre-print, Under review
    </div>
    <details class="paper-abstract">
      Dual process theory posits that human cognition arises via two systems. System 1, which is a quick, emotional, and intuitive process, which is subject to cognitive biases, and System 2, is a slow, onerous, and deliberate process. NLP researchers often compare zero-shot prompting in LLMs to System 1 reasoning and chain-of-thought (CoT) prompting to System 2. In line with this interpretation, prior research has found that using CoT prompting in LLMs leads to reduced gender bias. We investigate the relationship between bias, CoT prompting, a debiasing prompt, and dual process theory in LLMs directly. We compare zero-shot CoT, debiasing, and a variety of dual process theory-based prompting strategies on two bias datasets spanning nine different social bias categories. We incorporate human and machine personas to determine whether the effects of dual process theory in LLMs exist independent of explicit persona models or are based on modeling human cognition. We find that a human persona, debiasing, System 2, and CoT prompting all tend to reduce social biases in LLMs, though the best combination of features depends on the exact model and bias category -- resulting in up to a 19 percent drop in stereotypical judgments by an LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14644v1">zsLLMCode: An Effective Approach for Functional Code Embedding via LLM with Zero-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2024-09-23
    </div>
    <details class="paper-abstract">
      Regarding software engineering (SE) tasks, Large language models (LLMs) have the capability of zero-shot learning, which does not require training or fine-tuning, unlike pre-trained models (PTMs). However, LLMs are primarily designed for natural language output, and cannot directly produce intermediate embeddings from source code. They also face some challenges, for example, the restricted context length may prevent them from handling larger inputs, limiting their applicability to many SE tasks; while hallucinations may occur when LLMs are applied to complex downstream tasks. Motivated by the above facts, we propose zsLLMCode, a novel approach that generates functional code embeddings using LLMs. Our approach utilizes LLMs to convert source code into concise summaries through zero-shot learning, which is then transformed into functional code embeddings using specialized embedding models. This unsupervised approach eliminates the need for training and addresses the issue of hallucinations encountered with LLMs. To the best of our knowledge, this is the first approach that combines LLMs and embedding models to generate code embeddings. We conducted experiments to evaluate the performance of our approach. The results demonstrate the effectiveness and superiority of our approach over state-of-the-art unsupervised methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14572v1">Evaluating the Performance and Robustness of LLMs in Materials Science Q&A and Property Predictions</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to revolutionize scientific research, yet their robustness and reliability in domain-specific applications remain insufficiently explored. This study conducts a comprehensive evaluation and robustness analysis of LLMs within the field of materials science, focusing on domain-specific question answering and materials property prediction. Three distinct datasets are used in this study: 1) a set of multiple-choice questions from undergraduate-level materials science courses, 2) a dataset including various steel compositions and yield strengths, and 3) a band gap dataset, containing textual descriptions of material crystal structures and band gap values. The performance of LLMs is assessed using various prompting strategies, including zero-shot chain-of-thought, expert prompting, and few-shot in-context learning. The robustness of these models is tested against various forms of 'noise', ranging from realistic disturbances to intentionally adversarial manipulations, to evaluate their resilience and reliability under real-world conditions. Additionally, the study uncovers unique phenomena of LLMs during predictive tasks, such as mode collapse behavior when the proximity of prompt examples is altered and performance enhancement from train/test mismatch. The findings aim to provide informed skepticism for the broad use of LLMs in materials science and to inspire advancements that enhance their robustness and reliability for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16145v1">Learning to Localize Actions in Instructional Videos with LLM-Based Multi-Pathway Text-Video Alignment</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 Accepted to ECCV 2024
    </div>
    <details class="paper-abstract">
      Learning to localize temporal boundaries of procedure steps in instructional videos is challenging due to the limited availability of annotated large-scale training videos. Recent works focus on learning the cross-modal alignment between video segments and ASR-transcripted narration texts through contrastive learning. However, these methods fail to account for the alignment noise, i.e., irrelevant narrations to the instructional task in videos and unreliable timestamps in narrations. To address these challenges, this work proposes a novel training framework. Motivated by the strong capabilities of Large Language Models (LLMs) in procedure understanding and text summarization, we first apply an LLM to filter out task-irrelevant information and summarize task-related procedure steps (LLM-steps) from narrations. To further generate reliable pseudo-matching between the LLM-steps and the video for training, we propose the Multi-Pathway Text-Video Alignment (MPTVA) strategy. The key idea is to measure alignment between LLM-steps and videos via multiple pathways, including: (1) step-narration-video alignment using narration timestamps, (2) direct step-to-video alignment based on their long-term semantic similarity, and (3) direct step-to-video alignment focusing on short-term fine-grained semantic similarity learned from general video domains. The results from different pathways are fused to generate reliable pseudo step-video matching. We conducted extensive experiments across various tasks and problem settings to evaluate our proposed method. Our approach surpasses state-of-the-art methods in three downstream tasks: procedure step grounding, step localization, and narration grounding by 5.9\%, 3.1\%, and 2.8\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14506v1">InteLiPlan: Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      We introduce a lightweight LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting onboard embodied intelligence. By addressing challenges such as kinematic constraints and dynamic environments, our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline. Our framework, InteLiPlan, ensures that the LLM model's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention in the case where the system fails. We evaluate our method in both simulation and on the real Toyota HSR robot. The results show that our method achieves a 93% success rate in the fetch me task completion with system failure recovery, outperforming the baseline method in a domestic environment. InteLiPlan achieves comparable performance to the state-of-the-art large-scale LLM-based robotics planner, while guaranteeing real-time onboard computing with embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14488v1">Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      There is a growing interest in integrating Large Language Models (LLMs) with autonomous driving (AD) systems. However, AD systems are vulnerable to attacks against their object detection and tracking (ODT) functions. Unfortunately, our evaluation of four recent LLM agents against ODT attacks shows that the attacks are 63.26% successful in causing them to crash or violate traffic rules due to (1) misleading memory modules that provide past experiences for decision making, (2) limitations of prompts in identifying inconsistencies, and (3) reliance on ground truth perception data. In this paper, we introduce Hudson, a driving reasoning agent that extends prior LLM-based driving systems to enable safer decision making during perception attacks while maintaining effectiveness under benign conditions. Hudson achieves this by first instrumenting the AD software to collect real-time perception results and contextual information from the driving scene. This data is then formalized into a domain-specific language (DSL). To guide the LLM in detecting and making safe control decisions during ODT attacks, Hudson translates the DSL into natural language, along with a list of custom attack detection instructions. Following query execution, Hudson analyzes the LLM's control decision to understand its causal reasoning process. We evaluate the effectiveness of Hudson using a proprietary LLM (GPT-4) and two open-source LLMs (Llama and Gemma) in various adversarial driving scenarios. GPT-4, Llama, and Gemma achieve, on average, an attack detection accuracy of 83. 3%, 63. 6%, and 73. 6%. Consequently, they make safe control decisions in 86.4%, 73.9%, and 80% of the attacks. Our results, following the growing interest in integrating LLMs into AD systems, highlight the strengths of LLMs and their potential to detect and mitigate ODT attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14899v3">Stop Reasoning! When Multimodal LLM with Chain-of-Thought Reasoning Meets Adversarial Image</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) with a great ability of text and image understanding have received great attention. To achieve better reasoning with MLLMs, Chain-of-Thought (CoT) reasoning has been widely explored, which further promotes MLLMs' explainability by giving intermediate reasoning steps. Despite the strong power demonstrated by MLLMs in multimodal reasoning, recent studies show that MLLMs still suffer from adversarial images. This raises the following open questions: Does CoT also enhance the adversarial robustness of MLLMs? What do the intermediate reasoning steps of CoT entail under adversarial attacks? To answer these questions, we first generalize existing attacks to CoT-based inferences by attacking the two main components, i.e., rationale and answer. We find that CoT indeed improves MLLMs' adversarial robustness against the existing attack methods by leveraging the multi-step reasoning process, but not substantially. Based on our findings, we further propose a novel attack method, termed as stop-reasoning attack, that attacks the model while bypassing the CoT reasoning process. Experiments on three MLLMs and two visual reasoning datasets verify the effectiveness of our proposed method. We show that stop-reasoning attack can result in misled predictions and outperform baseline attacks by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14469v1">Rethinking Semantic Parsing for Large Language Models: Enhancing LLM Performance with Semantic Hints</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Semantic Parsing aims to capture the meaning of a sentence and convert it into a logical, structured form. Previous studies show that semantic parsing enhances the performance of smaller models (e.g., BERT) on downstream tasks. However, it remains unclear whether the improvements extend similarly to LLMs. In this paper, our empirical findings reveal that, unlike smaller models, directly adding semantic parsing results into LLMs reduces their performance. To overcome this, we propose SENSE, a novel prompting approach that embeds semantic hints within the prompt. Experiments show that SENSE consistently improves LLMs' performance across various tasks, highlighting the potential of integrating semantic information to improve LLM capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14436v1">Automotive innovation landscaping using LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 9pages, 4Figures, 1 Flow chart
    </div>
    <details class="paper-abstract">
      The process of landscaping automotive innovation through patent analysis is crucial for Research and Development teams. It aids in comprehending innovation trends, technological advancements, and the latest technologies from competitors. Traditionally, this process required intensive manual efforts. However, with the advent of Large Language Models (LLMs), it can now be automated, leading to faster and more efficient patent categorization & state-of-the-art of inventive concept extraction. This automation can assist various R\&D teams in extracting relevant information from extensive patent databases. This paper introduces a method based on prompt engineering to extract essential information for landscaping. The information includes the problem addressed by the patent, the technology utilized, and the area of innovation within the vehicle ecosystem (such as safety, Advanced Driver Assistance Systems and more).The result demonstrates the implementation of this method to create a landscape of fuel cell technology using open-source patent data. This approach provides a comprehensive overview of the current state of fuel cell technology, offering valuable insights for future research and development in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04197v2">DICE: Detecting In-distribution Contamination in LLM's Fine-tuning Phase for Math Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 13 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) relies on evaluation using public benchmarks, but data contamination can lead to overestimated performance. Previous researches focus on detecting contamination by determining whether the model has seen the exact same data during training. Besides, prior work has already shown that even training on data similar to benchmark data inflates performance, namely \emph{In-distribution contamination}. In this work, we argue that in-distribution contamination can lead to the performance drop on OOD benchmarks. To effectively detect in-distribution contamination, we propose DICE, a novel method that leverages the internal states of LLMs to locate-then-detect the contamination. DICE first identifies the most sensitive layer to contamination, then trains a classifier based on the internal states of that layer. Experiments reveal DICE's high accuracy in detecting in-distribution contamination across various LLMs and math reasoning datasets. We also show the generalization capability of the trained DICE detector, which is able to detect contamination across multiple benchmarks with similar distributions. Additionally, we find that DICE's predictions correlate with the performance of LLMs fine-tuned by either us or other organizations, achieving a coefficient of determination ($R^2$) between 0.61 and 0.75. The code and data are available at https://github.com/THU-KEG/DICE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14329v1">ISC4DGF: Enhancing Directed Grey-box Fuzzing with LLM-Driven Initial Seed Corpus Generation</a></div>
    <div class="paper-meta">
      📅 2024-09-22
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Fuzz testing is crucial for identifying software vulnerabilities, with coverage-guided grey-box fuzzers like AFL and Angora excelling in broad detection. However, as the need for targeted detection grows, directed grey-box fuzzing (DGF) has become essential, focusing on specific vulnerabilities. The initial seed corpus, which consists of carefully selected input samples that the fuzzer uses as a starting point, is fundamental in determining the paths that the fuzzer explores. A well-designed seed corpus can guide the fuzzer more effectively towards critical areas of the code, improving the efficiency and success of the fuzzing process. Even with its importance, many works concentrate on refining guidance mechanisms while paying less attention to optimizing the initial seed corpus. In this paper, we introduce ISC4DGF, a novel approach to generating optimized initial seed corpus for DGF using Large Language Models (LLMs). By leveraging LLMs' deep software understanding and refined user inputs, ISC4DGF creates precise seed corpus that efficiently trigger specific vulnerabilities. Implemented on AFL and tested against state-of-the-art fuzzers like AFLGo, FairFuzz, and Entropic using the Magma benchmark, ISC4DGF achieved a 35.63x speedup and 616.10x fewer target reaches. Moreover, ISC4DGF focused on more effectively detecting target vulnerabilities, enhancing efficiency while operating with reduced code coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.09895v6">Knowledge Transfer from High-Resource to Low-Resource Programming Languages for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      Over the past few years, Large Language Models of Code (Code LLMs) have started to have a significant impact on programming practice. Code LLMs are also emerging as building blocks for research in programming languages and software engineering. However, Code LLMs produce impressive results on programming languages that are well represented in their training data (e.g., Java, Python, or JavaScript), but struggle with low-resource languages that have limited training data available. Low resource languages include OCaml, Racket, and several others. This paper presents an effective approach for boosting the performance of Code LLMs on low-resource languages using semi-synthetic data. Our approach, MultiPL-T, translates training data from high-resource languages into training data for low-resource languages in the following way. 1) We use a Code LLM to synthesize tests for commented code from a high-resource language, filtering out faulty tests and code with low test coverage. 2) We use a Code LLM to translate Python code to a target low-resource language, and use tests to validate the translation. We apply this approach to generate tens of thousands of validated training items for Julia, Lua, OCaml, R, and Racket. Furthermore, we use an open model (StarCoderBase) with open training data (The Stack), which allows us to decontaminate benchmarks, train models without violating licenses, and run experiments that could not otherwise be done. With MultiPL-T generated data, we present fine-tuned versions of StarCoderBase and Code Llama for Julia, Lua, OCaml, R, and Racket. On established benchmarks (MultiPL-E), these models outperform other open Code LLMs. The MultiPL-T approach is easy to apply to new languages, and is significantly more efficient and effective than alternatives such as training longer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14306v1">LLMs are One-Shot URL Classifiers and Explainers</a></div>
    <div class="paper-meta">
      📅 2024-09-22
    </div>
    <details class="paper-abstract">
      Malicious URL classification represents a crucial aspect of cyber security. Although existing work comprises numerous machine learning and deep learning-based URL classification models, most suffer from generalisation and domain-adaptation issues arising from the lack of representative training datasets. Furthermore, these models fail to provide explanations for a given URL classification in natural human language. In this work, we investigate and demonstrate the use of Large Language Models (LLMs) to address this issue. Specifically, we propose an LLM-based one-shot learning framework that uses Chain-of-Thought (CoT) reasoning to predict whether a given URL is benign or phishing. We evaluate our framework using three URL datasets and five state-of-the-art LLMs and show that one-shot LLM prompting indeed provides performances close to supervised models, with GPT 4-Turbo being the best model, followed by Claude 3 Opus. We conduct a quantitative analysis of the LLM explanations and show that most of the explanations provided by LLMs align with the post-hoc explanations of the supervised classifiers, and the explanations have high readability, coherency, and informativeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14195v1">The Imperative of Conversation Analysis in the Era of LLMs: A Survey of Tasks, Techniques, and Trends</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 21 pages, work in progress
    </div>
    <details class="paper-abstract">
      In the era of large language models (LLMs), a vast amount of conversation logs will be accumulated thanks to the rapid development trend of language UI. Conversation Analysis (CA) strives to uncover and analyze critical information from conversation data, streamlining manual processes and supporting business insights and decision-making. The need for CA to extract actionable insights and drive empowerment is becoming increasingly prominent and attracting widespread attention. However, the lack of a clear scope for CA leads to a dispersion of various techniques, making it difficult to form a systematic technical synergy to empower business applications. In this paper, we perform a thorough review and systematize CA task to summarize the existing related work. Specifically, we formally define CA task to confront the fragmented and chaotic landscape in this field, and derive four key steps of CA from conversation scene reconstruction, to in-depth attribution analysis, and then to performing targeted training, finally generating conversations based on the targeted training for achieving the specific goals. In addition, we showcase the relevant benchmarks, discuss potential challenges and point out future directions in both industry and academia. In view of current advancements, it is evident that the majority of efforts are still concentrated on the analysis of shallow conversation elements, which presents a considerable gap between the research and business, and with the assist of LLMs, recent work has shown a trend towards research on causality and strategic tasks which are sophisticated and high-level. The analyzed experiences and insights will inevitably have broader application value in business operations that target conversation logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14175v1">QMOS: Enhancing LLMs for Telecommunication with Question Masked loss and Option Shuffling</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have brought about substantial advancements in the field of Question Answering (QA) systems. These models do remarkably well in addressing intricate inquiries in a variety of disciplines. However, because of domain-specific vocabulary, complex technological concepts, and the requirement for exact responses applying LLMs to specialized sectors like telecommunications presents additional obstacles. GPT-3.5 has been used in recent work, to obtain noteworthy accuracy for telecom-related questions in a Retrieval Augmented Generation (RAG) framework. Notwithstanding these developments, the practical use of models such as GPT-3.5 is restricted by their proprietary nature and high computing demands. This paper introduces QMOS, an innovative approach which uses a Question-Masked loss and Option Shuffling trick to enhance the performance of LLMs in answering Multiple-Choice Questions in the telecommunications domain. Our focus was on using opensource, smaller language models (Phi-2 and Falcon-7B) within an enhanced RAG framework. Our multi-faceted approach involves several enhancements to the whole LLM-RAG pipeline of finetuning, retrieval, prompt engineering and inference. Our approaches significantly outperform existing results, achieving accuracy improvements from baselines of 24.70% to 49.30% with Falcon-7B and from 42.07% to 84.65% with Phi-2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09629v2">Confidence Estimation for LLM-Based Dialogue State Tracking</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 Accepted for publication at IEEE SLT 2024
    </div>
    <details class="paper-abstract">
      Estimation of a model's confidence on its outputs is critical for Conversational AI systems based on large language models (LLMs), especially for reducing hallucination and preventing over-reliance. In this work, we provide an exhaustive exploration of methods, including approaches proposed for open- and closed-weight LLMs, aimed at quantifying and leveraging model uncertainty to improve the reliability of LLM-generated responses, specifically focusing on dialogue state tracking (DST) in task-oriented dialogue systems (TODS). Regardless of the model type, well-calibrated confidence scores are essential to handle uncertainties, thereby improving model performance. We evaluate four methods for estimating confidence scores based on softmax, raw token scores, verbalized confidences, and a combination of these methods, using the area under the curve (AUC) metric to assess calibration, with higher AUC indicating better calibration. We also enhance these with a self-probing mechanism, proposed for closed models. Furthermore, we assess these methods using an open-weight model fine-tuned for the task of DST, achieving superior joint goal accuracy (JGA). Our findings also suggest that fine-tuning open-weight LLMs can result in enhanced AUC performance, indicating better confidence score calibration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14082v1">PTD-SQL: Partitioning and Targeted Drilling with LLMs in Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 EMNLP 2024 Main Conference. Revised by ARR April and ARR June. 32 pages, 7 figures and 30 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for Text-to-SQL tasks, exhibiting remarkable reasoning capabilities. Different from tasks such as math word problems and commonsense reasoning, SQL solutions have a relatively fixed pattern. This facilitates the investigation of whether LLMs can benefit from categorical thinking, mirroring how humans acquire knowledge through inductive reasoning based on comparable examples. In this study, we propose that employing query group partitioning allows LLMs to focus on learning the thought processes specific to a single problem type, consequently enhancing their reasoning abilities across diverse difficulty levels and problem categories. Our experiments reveal that multiple advanced LLMs, when equipped with PTD-SQL, can either surpass or match previous state-of-the-art (SOTA) methods on the Spider and BIRD datasets. Intriguingly, models with varying initial performances have exhibited significant improvements, mainly at the boundary of their capabilities after targeted drilling, suggesting a parallel with human progress. Code is available at https://github.com/lrlbbzl/PTD-SQL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06799v2">LLM-dCache: Improving Tool-Augmented LLMs with GPT-Driven Localized Data Caching</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 ICECS 2024 Camera-Ready
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) broaden their capabilities to manage thousands of API calls, they are confronted with complex data operations across vast datasets with significant overhead to the underlying system. In this work, we introduce LLM-dCache to optimize data accesses by treating cache operations as callable API functions exposed to the tool-augmented agent. We grant LLMs the autonomy to manage cache decisions via prompting, seamlessly integrating with existing function-calling mechanisms. Tested on an industry-scale massively parallel platform that spans hundreds of GPT endpoints and terabytes of imagery, our method improves Copilot times by an average of 1.24x across various LLMs and prompting techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14058v1">Practically implementing an LLM-supported collaborative vulnerability remediation process: a team-based approach</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 accepted by Computers & Security
    </div>
    <details class="paper-abstract">
      Incorporating LLM into cybersecurity operations, a typical real-world high-stakes task, is critical but non-trivial in practice. Using cybersecurity as the study context, we conduct a three-step mix-method study to incorporate LLM into the vulnerability remediation process effectively. Specifically, we deconstruct the deficiencies in user satisfaction within the existing process (Study 1). This inspires us to design, implement, and empirically validate an LLM-supported collaborative vulnerability remediation process through a field study (Study 2). Given LLM's diverse contributions, we further investigate LLM's double-edge roles through the analysis of remediation reports and follow-up interviews (Study 3). In essence, our contribution lies in promoting an efficient LLM-supported collaborative vulnerability remediation process. These first-hand, real-world pieces of evidence suggest that when incorporating LLMs into practical processes, facilitating the collaborations among all associated stakeholders, reshaping LLMs' roles according to task complexity, as well as approaching the short-term side effects of improved user engagement facilitated by LLMs with a rational mindset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16437v3">Reasoning Runtime Behavior of a Program with LLM: How Far Are We?</a></div>
    <div class="paper-meta">
      📅 2024-09-21
      | 💬 Accepted by 47th IEEE/ACM International Conference on Software Engineering (ICSE 2025). Our REval leaderboard is available at https://r-eval.github.io
    </div>
    <details class="paper-abstract">
      Large language models for code (i.e., code LLMs) have shown strong code understanding and generation capabilities. To evaluate the capabilities of code LLMs in various aspects, many benchmarks have been proposed (e.g., HumanEval and ClassEval). Code reasoning is one of the most essential abilities of code LLMs, but existing benchmarks for code reasoning are not sufficient. Typically, they focus on predicting the input and output of a program, ignoring the evaluation of the intermediate behavior during program execution, as well as the logical consistency (e.g., the model should not give the correct output if the prediction of execution path is wrong) when performing the reasoning. To address these problems, in this paper, we propose a framework, namely REval, for evaluating code reasoning abilities and consistency of code LLMs with program execution. We utilize existing code benchmarks and adapt them to new benchmarks within our framework. A large-scale empirical study is conducted and most LLMs show unsatisfactory performance on both Runtime Behavior Reasoning (i.e., an average accuracy of 44.4%) and Incremental Consistency Evaluation (i.e., an average IC score of 10.3). Evaluation results of current code LLMs reflect the urgent need for the community to strengthen the code reasoning capability of code LLMs. Our code, data, and \newname leaderboard are available at https://r-eval.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.19902v2">Herd: Using multiple, smaller LLMs to match the performances of proprietary, large LLMs via an intelligent composer</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      Currently, over a thousand LLMs exist that are multi-purpose and are capable of performing real world tasks, including Q&A, text summarization, content generation, etc. However, accessibility, scale and reliability of free models prevents them from being widely deployed in everyday use cases. To address the first two issues of access and scale, organisations such as HuggingFace have created model repositories where users have uploaded model weights and quantized versions of models trained using different paradigms, as well as model cards describing their training process. While some models report performance on commonly used benchmarks, not all do, and interpreting the real world impact of trading off performance on a benchmark for model deployment cost, is unclear. Here, we show that a herd of open source models can match or exceed the performance of proprietary models via an intelligent router. We show that a Herd of open source models is able to match the accuracy of ChatGPT, despite being composed of models that are effectively 2.5x smaller. We show that in cases where GPT is not able to answer the query, Herd is able to identify a model that can, at least 40% of the time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14037v1">Can LLMs replace Neil deGrasse Tyson? Evaluating the Reliability of LLMs as Science Communicators</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and AI assistants driven by these models are experiencing exponential growth in usage among both expert and amateur users. In this work, we focus on evaluating the reliability of current LLMs as science communicators. Unlike existing benchmarks, our approach emphasizes assessing these models on scientific questionanswering tasks that require a nuanced understanding and awareness of answerability. We introduce a novel dataset, SCiPS-QA, comprising 742 Yes/No queries embedded in complex scientific concepts, along with a benchmarking suite that evaluates LLMs for correctness and consistency across various criteria. We benchmark three proprietary LLMs from the OpenAI GPT family and 13 open-access LLMs from the Meta Llama-2, Llama-3, and Mistral families. While most open-access models significantly underperform compared to GPT-4 Turbo, our experiments identify Llama-3-70B as a strong competitor, often surpassing GPT-4 Turbo in various evaluation aspects. We also find that even the GPT models exhibit a general incompetence in reliably verifying LLM responses. Moreover, we observe an alarming trend where human evaluators are deceived by incorrect responses from GPT-4 Turbo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03688v1">LLM Agents as 6G Orchestrator: A Paradigm for Task-Oriented Physical-Layer Automation</a></div>
    <div class="paper-meta">
      📅 2024-09-21
    </div>
    <details class="paper-abstract">
      The rapid advancement in generative pre-training models is propelling a paradigm shift in technological progression from basic applications such as chatbots towards more sophisticated agent-based systems. It is with huge potential and necessity that the 6G system be combined with the copilot of large language model (LLM) agents and digital twins (DT) to manage the highly complicated communication system with new emerging features such as native AI service and sensing. With the 6G-oriented agent, the base station could understand the transmission requirements of various dynamic upper-layer tasks, automatically orchestrate the optimal system workflow. Through continuously get feedback from the 6G DT for reinforcement, the agents can finally raise the performance of practical system accordingly. Differing from existing LLM agents designed for general application, the 6G-oriented agent aims to make highly rigorous and precise planning with a vast amount of extra expert knowledge, which inevitably requires a specific system design from model training to implementation. This paper proposes a novel comprehensive approach for building task-oriented 6G LLM agents. We first propose a two-stage continual pre-training and fine-tuning scheme to build the field basic model and diversities of specialized expert models for meeting the requirements of various application scenarios. Further, a novel inference framework based on semantic retrieval for leveraging the existing communication-related functions is proposed. Experiment results of exemplary tasks, such as physical-layer task decomposition, show the proposed paradigm's feasibility and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13949v1">Mufu: Multilingual Fused Learning for Low-Resource Translation with LLM</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) are great translators, but this is largely limited to high-resource languages. For many LLMs, translating in and out of low-resource languages remains a challenging task. To maximize data efficiency in this low-resource setting, we introduce Mufu, which includes a selection of automatically generated multilingual candidates and an instruction to correct inaccurate translations in the prompt. Mufu prompts turn a translation task into a postediting one, and seek to harness the LLM's reasoning capability with auxiliary translation candidates, from which the model is required to assess the input quality, align the semantics cross-lingually, copy from relevant inputs and override instances that are incorrect. Our experiments on En-XX translations over the Flores-200 dataset show LLMs finetuned against Mufu-style prompts are robust to poor quality auxiliary translation candidates, achieving performance superior to NLLB 1.3B distilled model in 64% of low- and very-low-resource language pairs. We then distill these models to reduce inference cost, while maintaining on average 3.1 chrF improvement over finetune-only baseline in low-resource translations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13897v1">LLM for Everyone: Representing the Underrepresented in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 PhD thesis
    </div>
    <details class="paper-abstract">
      Natural language processing (NLP) has witnessed a profound impact of large language models (LLMs) that excel in a multitude of tasks. However, the limitation of LLMs in multilingual settings, particularly in underrepresented languages, remains a significant hurdle. This thesis aims to bridge the gap in NLP research and development by focusing on underrepresented languages. A comprehensive evaluation of LLMs is conducted to assess their capabilities in these languages, revealing the challenges of multilingual and multicultural generalization. Addressing the multilingual generalization gap, this thesis proposes data-and-compute-efficient methods to mitigate the disparity in LLM ability in underrepresented languages, allowing better generalization on underrepresented languages without the loss of task generalization ability. The proposed solutions cover cross-lingual continual instruction tuning, retrieval-based cross-lingual in-context learning, and in-context query alignment. Furthermore, a novel method to measure cultural values alignment between LLMs operating in different languages is proposed, ensuring cultural sensitivity and inclusivity. These contributions aim to enhance the multilingual and multicultural alignment of LLMs in underrepresented languages, ultimately advancing the NLP field toward greater equality and inclusiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11875v2">Concept Induction using LLMs: a user experiment for assessment</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Explainable Artificial Intelligence (XAI) poses a significant challenge in providing transparent and understandable insights into complex AI models. Traditional post-hoc algorithms, while useful, often struggle to deliver interpretable explanations. Concept-based models offer a promising avenue by incorporating explicit representations of concepts to enhance interpretability. However, existing research on automatic concept discovery methods is often limited by lower-level concepts, costly human annotation requirements, and a restricted domain of background knowledge. In this study, we explore the potential of a Large Language Model (LLM), specifically GPT-4, by leveraging its domain knowledge and common-sense capability to generate high-level concepts that are meaningful as explanations for humans, for a specific setting of image classification. We use minimal textual object information available in the data via prompting to facilitate this process. To evaluate the output, we compare the concepts generated by the LLM with two other methods: concepts generated by humans and the ECII heuristic concept induction system. Since there is no established metric to determine the human understandability of concepts, we conducted a human study to assess the effectiveness of the LLM-generated concepts. Our findings indicate that while human-generated explanations remain superior, concepts derived from GPT-4 are more comprehensible to humans compared to those generated by ECII.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13884v1">A Multi-LLM Debiasing Framework</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful tools with the potential to benefit society immensely, yet, they have demonstrated biases that perpetuate societal inequalities. Despite significant advancements in bias mitigation techniques using data augmentation, zero-shot prompting, and model fine-tuning, biases continuously persist, including subtle biases that may elude human detection. Recent research has shown a growing interest in multi-LLM approaches, which have been demonstrated to be effective in improving the quality of reasoning and factuality in LLMs. Building on this approach, we propose a novel multi-LLM debiasing framework aimed at reducing bias in LLMs. Our work is the first to introduce and evaluate two distinct approaches within this framework for debiasing LLMs: a centralized method, where the conversation is facilitated by a single central LLM, and a decentralized method, where all models communicate directly. Our findings reveal that our multi-LLM framework significantly reduces bias in LLMs, outperforming the baseline method across several social groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.08614v2">XplainLLM: A Knowledge-Augmented Dataset for Reliable Grounded Explanations in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 Accepted at EMNLP 2024 Main. The dataset and code are available at https://github.com/chen-zichen/XplainLLM_dataset.git
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in natural language tasks, yet understanding their reasoning processes remains a significant challenge. We address this by introducing XplainLLM, a dataset accompanying an explanation framework designed to enhance LLM transparency and reliability. Our dataset comprises 24,204 instances where each instance interprets the LLM's reasoning behavior using knowledge graphs (KGs) and graph attention networks (GAT), and includes explanations of LLMs such as the decoder-only Llama-3 and the encoder-only RoBERTa. XplainLLM also features a framework for generating grounded explanations and the debugger-scores for multidimensional quality analysis. Our explanations include why-choose and why-not-choose components, reason-elements, and debugger-scores that collectively illuminate the LLM's reasoning behavior. Our evaluations demonstrate XplainLLM's potential to reduce hallucinations and improve grounded explanation generation in LLMs. XplainLLM is a resource for researchers and practitioners to build trust and verify the reliability of LLM outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13852v1">Do language models practice what they preach? Examining language ideologies about gendered language reform encoded in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      We study language ideologies in text produced by LLMs through a case study on English gendered language reform (related to role nouns like congressperson/-woman/-man, and singular they). First, we find political bias: when asked to use language that is "correct" or "natural", LLMs use language most similarly to when asked to align with conservative (vs. progressive) values. This shows how LLMs' metalinguistic preferences can implicitly communicate the language ideologies of a particular political group, even in seemingly non-political contexts. Second, we find LLMs exhibit internal inconsistency: LLMs use gender-neutral variants more often when more explicit metalinguistic context is provided. This shows how the language ideologies expressed in text produced by LLMs can vary, which may be unexpected to users. We discuss the broader implications of these findings for value alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13642v1">Enhancing Fault Localization Through Ordered Code Analysis with LLM Agents and Self-Reflection</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Locating and fixing software faults is a time-consuming and resource-intensive task in software development. Traditional fault localization methods, such as Spectrum-Based Fault Localization (SBFL), rely on statistical analysis of test coverage data but often suffer from lower accuracy. Learning-based techniques, while more effective, require extensive training data and can be computationally expensive. Recent advancements in Large Language Models (LLMs) offer promising improvements in fault localization by enhancing code comprehension and reasoning. However, these LLM-based techniques still face challenges, including token limitations, degraded performance with long inputs, and difficulties managing large-scale projects with complex systems involving multiple interacting components. To address these issues, we introduce LLM4FL, a novel LLM-agent-based fault localization approach that integrates SBFL rankings with a divide-and-conquer strategy. By dividing large coverage data into manageable groups and employing multiple LLM agents through prompt chaining, LLM4FL navigates the codebase and localizes faults more effectively. The approach also incorporates self-reflection and chain-of-thought reasoning, enabling agents to iteratively generate fixes and re-rank suspicious methods. We evaluated LLM4FL on the Defects4J (V2.0.0) benchmark, comprising 675 real-world faults from 14 open-source Java projects. Our results demonstrate that LLM4FL outperforms AutoFL by 19.27% in Top-1 accuracy and surpasses state-of-the-art supervised techniques such as DeepFL and Grace, all without task-specific training. Additionally, we highlight the impact of coverage splitting and prompt chaining on fault localization performance and show that different method ordering can improve Top-1 accuracy by up to 22%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13588v1">ChainBuddy: An AI Agent System for Generating LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 12 pages, 5 figures, pre-print
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) advance, their potential applications have grown significantly. However, it remains difficult to evaluate LLM behavior on user-specific tasks and craft effective pipelines to do so. Many users struggle with where to start, often referred to as the "blank page" problem. ChainBuddy, an AI assistant for generating evaluative LLM pipelines built into the ChainForge platform, aims to tackle this issue. ChainBuddy offers a straightforward and user-friendly way to plan and evaluate LLM behavior, making the process less daunting and more accessible across a wide range of possible tasks and use cases. We report a within-subjects user study comparing ChainBuddy to the baseline interface. We find that when using AI assistance, participants reported a less demanding workload and felt more confident setting up evaluation pipelines of LLM behavior. We derive insights for the future of interfaces that assist users in the open-ended evaluation of AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.15676v2">Beyond Chain-of-Thought: A Survey of Chain-of-X Paradigms for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been a widely adopted prompting method, eliciting impressive reasoning abilities of Large Language Models (LLMs). Inspired by the sequential thought structure of CoT, a number of Chain-of-X (CoX) methods have been developed to address various challenges across diverse domains and tasks involving LLMs. In this paper, we provide a comprehensive survey of Chain-of-X methods for LLMs in different contexts. Specifically, we categorize them by taxonomies of nodes, i.e., the X in CoX, and application tasks. We also discuss the findings and implications of existing CoX methods, as well as potential future directions. Our survey aims to serve as a detailed and up-to-date resource for researchers seeking to apply the idea of CoT to broader scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13524v1">Contextualized AI for Cyber Defense: An Automated Survey using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 8 pages, 2 figures, 4 tables, accepted into 17th International Conference on Security of Information and Networks (SINCONF 2024)
    </div>
    <details class="paper-abstract">
      This paper surveys the potential of contextualized AI in enhancing cyber defense capabilities, revealing significant research growth from 2015 to 2024. We identify a focus on robustness, reliability, and integration methods, while noting gaps in organizational trust and governance frameworks. Our study employs two LLM-assisted literature survey methodologies: (A) ChatGPT 4 for exploration, and (B) Gemma 2:9b for filtering with Claude 3.5 Sonnet for full-text analysis. We discuss the effectiveness and challenges of using LLMs in academic research, providing insights for future researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13484v1">'Since Lawyers are Males..': Examining Implicit Gender Bias in Hindi Language Generation by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to generate text across various languages, for tasks such as translation, customer support, and education. Despite these advancements, LLMs show notable gender biases in English, which become even more pronounced when generating content in relatively underrepresented languages like Hindi. This study explores implicit gender biases in Hindi text generation and compares them to those in English. We developed Hindi datasets inspired by WinoBias to examine stereotypical patterns in responses from models like GPT-4o and Claude-3 sonnet. Our results reveal a significant gender bias of 87.8% in Hindi, compared to 33.4% in English GPT-4o generation, with Hindi responses frequently relying on gender stereotypes related to occupations, power hierarchies, and social class. This research underscores the variation in gender biases across languages and provides considerations for navigating these biases in generative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11871v3">Generative AI Voting: Fair Collective Choice is Resilient to LLM Biases and Inconsistencies</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 33 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Scaling up deliberative and voting participation is a longstanding endeavor -- a cornerstone for direct democracy and legitimate collective choice. Recent breakthroughs in generative artificial intelligence (AI) and large language models (LLMs) unravel new capabilities for AI personal assistants to overcome cognitive bandwidth limitations of humans, providing decision support or even direct representation of human voters at large scale. However, the quality of this representation and what underlying biases manifest when delegating collective decision-making to LLMs is an alarming and timely challenge to tackle. By rigorously emulating with high realism more than >50K LLM voting personas in 81 real-world voting elections, we disentangle the nature of different biases in LLMS (GPT 3, GPT 3.5, and Llama2). Complex preferential ballot formats exhibit significant inconsistencies compared to simpler majoritarian elections that show higher consistency. Strikingly though, by demonstrating for the first time in real-world a proportional representation of voters in direct democracy, we are also able to show that fair ballot aggregation methods, such as equal shares, prove to be a win-win: fairer voting outcomes for humans with fairer AI representation. This novel underlying relationship proves paramount for democratic resilience in progressives scenarios with low voters turnout and voter fatigue supported by AI representatives: abstained voters are mitigated by recovering highly representative voting outcomes that are fairer. These interdisciplinary insights provide remarkable foundations for science, policymakers, and citizens to develop safeguards and resilience for AI risks in democratic innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13373v1">LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      The ability to plan a course of action that achieves a desired state of affairs has long been considered a core competence of intelligent agents and has been an integral part of AI research since its inception. With the advent of large language models (LLMs), there has been considerable interest in the question of whether or not they possess such planning abilities. PlanBench, an extensible benchmark we developed in 2022, soon after the release of GPT3, has remained an important tool for evaluating the planning abilities of LLMs. Despite the slew of new private and open source LLMs since GPT3, progress on this benchmark has been surprisingly slow. OpenAI claims that their recent o1 (Strawberry) model has been specifically constructed and trained to escape the normal limitations of autoregressive LLMs--making it a new kind of model: a Large Reasoning Model (LRM). Using this development as a catalyst, this paper takes a comprehensive look at how well current LLMs and new LRMs do on PlanBench. As we shall see, while o1's performance is a quantum improvement on the benchmark, outpacing the competition, it is still far from saturating it. This improvement also brings to the fore questions about accuracy, efficiency, and guarantees which must be considered before deploying such systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13356v1">Automatic Behavior Tree Expansion with LLMs for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 Submitted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Robotic systems for manipulation tasks are increasingly expected to be easy to configure for new tasks or unpredictable environments, while keeping a transparent policy that is readable and verifiable by humans. We propose the method BEhavior TRee eXPansion with Large Language Models (BETR-XP-LLM) to dynamically and automatically expand and configure Behavior Trees as policies for robot control. The method utilizes an LLM to resolve errors outside the task planner's capabilities, both during planning and execution. We show that the method is able to solve a variety of tasks and failures and permanently update the policy to handle similar problems in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17174v1">CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Casual Significance and Consistency</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal illusions between \textit{a step of reasoning} and \textit{corresponding state transitions} are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13252v1">Leveraging Knowledge Graphs and LLMs to Support and Monitor Legislative Systems</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Knowledge Graphs (KGs) have been used to organize large datasets into structured, interconnected information, enhancing data analytics across various fields. In the legislative context, one potential natural application of KGs is modeling the intricate set of interconnections that link laws and their articles with each other and the broader legislative context. At the same time, the rise of large language models (LLMs) such as GPT has opened new opportunities in legal applications, such as text generation and document drafting. Despite their potential, the use of LLMs in legislative contexts is critical since it requires the absence of hallucinations and reliance on up-to-date information, as new laws are published on a daily basis. This work investigates how Legislative Knowledge Graphs and LLMs can synergize and support legislative processes. We address three key questions: the benefits of using KGs for legislative systems, how LLM can support legislative activities by ensuring an accurate output, and how we can allow non-technical users to use such technologies in their activities. To this aim, we develop Legis AI Platform, an interactive platform focused on Italian legislation that enhances the possibility of conducting legislative analysis and that aims to support lawmaking activities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13177v1">An Adaptive End-to-End IoT Security Framework Using Explainable AI and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-20
      | 💬 6 pages, 1 figure, Accepted in 2024 IEEE WF-IoT Conference
    </div>
    <details class="paper-abstract">
      The exponential growth of the Internet of Things (IoT) has significantly increased the complexity and volume of cybersecurity threats, necessitating the development of advanced, scalable, and interpretable security frameworks. This paper presents an innovative, comprehensive framework for real-time IoT attack detection and response that leverages Machine Learning (ML), Explainable AI (XAI), and Large Language Models (LLM). By integrating XAI techniques such as SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) with a model-independent architecture, we ensure our framework's adaptability across various ML algorithms. Additionally, the incorporation of LLMs enhances the interpretability and accessibility of detection decisions, providing system administrators with actionable, human-understandable explanations of detected threats. Our end-to-end framework not only facilitates a seamless transition from model development to deployment but also represents a real-world application capability that is often lacking in existing research. Based on our experiments with the CIC-IOT-2023 dataset \cite{neto2023ciciot2023}, Gemini and OPENAI LLMS demonstrate unique strengths in attack mitigation: Gemini offers precise, focused strategies, while OPENAI provides extensive, in-depth security measures. Incorporating SHAP and LIME algorithms within XAI provides comprehensive insights into attack detection, emphasizing opportunities for model improvement through detailed feature analysis, fine-tuning, and the adaptation of misclassifications to enhance accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.13884v1">A Multi-LLM Debiasing Framework</a></div>
    <div class="paper-meta">
      📅 2024-09-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful tools with the potential to benefit society immensely, yet, they have demonstrated biases that perpetuate societal inequalities. Despite significant advancements in bias mitigation techniques using data augmentation, zero-shot prompting, and model fine-tuning, biases continuously persist, including subtle biases that may elude human detection. Recent research has shown a growing interest in multi-LLM approaches, which have been demonstrated to be effective in improving the quality of reasoning and factuality in LLMs. Building on this approach, we propose a novel multi-LLM debiasing framework aimed at reducing bias in LLMs. Our work is the first to introduce and evaluate two distinct approaches within this framework for debiasing LLMs: a centralized method, where the conversation is facilitated by a single central LLM, and a decentralized method, where all models communicate directly. Our findings reveal that our multi-LLM framework significantly reduces bias in LLMs, outperforming the baseline method across several social groups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13109v1">Visualizationary: Automating Design Feedback for Visualization Designers using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Interactive visualization editors empower people to author visualizations without writing code, but do not guide them in the art and craft of effective visual communication. In this paper, we explore the potential for using an off-the-shelf Large Language Model (LLM) to provide actionable and customized feedback to visualization designers. Our implementation, called VISUALIZATIONARY, showcases how ChatGPT can be used in this manner using two components: a preamble of visualization design guidelines and a suite of perceptual filters extracting salient metrics from a visualization image. We present findings from a longitudinal user study involving 13 visualization designers - 6 novices, 4 intermediate ones, and 3 experts - authoring a new visualization from scratch over the course of several days. Our results indicate that providing guidance in natural language using an LLM can aid even seasoned designers in refining their visualizations. All supplemental materials accompanying this paper are available at https://osf.io/v7hu8.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18841v3">Navigating LLM Ethics: Advancements, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      This study addresses ethical issues surrounding Large Language Models (LLMs) within the field of artificial intelligence. It explores the common ethical challenges posed by both LLMs and other AI systems, such as privacy and fairness, as well as ethical challenges uniquely arising from LLMs. It highlights challenges such as hallucination, verifiable accountability, and decoding censorship complexity, which are unique to LLMs and distinct from those encountered in traditional AI systems. The study underscores the need to tackle these complexities to ensure accountability, reduce biases, and enhance transparency in the influential role that LLMs play in shaping information dissemination. It proposes mitigation strategies and future directions for LLM ethics, advocating for interdisciplinary collaboration. It recommends ethical frameworks tailored to specific domains and dynamic auditing systems adapted to diverse contexts. This roadmap aims to guide responsible development and integration of LLMs, envisioning a future where ethical considerations govern AI advancements in society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17172v1">What Would You Ask When You First Saw $a^2+b^2=c^2$? Evaluating LLM on Curiosity-Driven Questioning</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can store a massive amount of knowledge, yet their potential to acquire new knowledge remains unknown. We propose a novel evaluation framework that evaluates this capability. This framework prompts LLMs to generate questions about a statement introducing scientific knowledge, simulating a curious person when facing the statement for the first time. We score the qualities of the generated questions, thereby evaluating the knowledge acquisition potential of the LLM. We apply controlled ablation studies to validate our scoring procedures. Additionally, we created a synthetic dataset consisting of 1101 statements in physics, chemistry, and maths with distinct levels of difficulties, 300 general knowledge statements, and 567 incorrect statements. Human evaluations were conducted to validate our model assessments, achieving an approximate weighted Cohen's kappa of 0.7 on all three metrics considered. We find that while large models like GPT-4 and Mistral 8x7b are adept at generating coherent and relevant questions, the smaller Phi-2 model is equally or more effective. This indicates that size does not solely determine a model's knowledge acquisition potential. The proposed framework quantifies a critical model capability that was commonly overlooked and opens up research opportunities for developing more knowledgeable AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13093v1">Guided Profile Generation Improves Personalization with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-09-19
      | 💬 EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      In modern commercial systems, including Recommendation, Ranking, and E-Commerce platforms, there is a trend towards improving customer experiences by incorporating Personalization context as input into Large Language Models (LLMs). However, LLMs often struggle to effectively parse and utilize sparse and complex personal context without additional processing or contextual enrichment, underscoring the need for more sophisticated context understanding mechanisms. In this work, we propose Guided Profile Generation (GPG), a general method designed to generate personal profiles in natural language. As is observed, intermediate guided profile generation enables LLMs to summarize, and extract the important, distinctive features from the personal context into concise, descriptive sentences, precisely tailoring their generation more closely to an individual's unique habits and preferences. Our experimental results show that GPG improves LLM's personalization ability across different tasks, for example, it increases 37% accuracy in predicting personal preference compared to directly feeding the LLMs with raw personal context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00031v1">Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Machine-learning technologies are seeing increased deployment in real-world market scenarios. In this work, we explore the strategic behaviors of large language models (LLMs) when deployed as autonomous agents in multi-commodity markets, specifically within Cournot competition frameworks. We examine whether LLMs can independently engage in anti-competitive practices such as collusion or, more specifically, market division. Our findings demonstrate that LLMs can effectively monopolize specific commodities by dynamically adjusting their pricing and resource allocation strategies, thereby maximizing profitability without direct human input or explicit collusion commands. These results pose unique challenges and opportunities for businesses looking to integrate AI into strategic roles and for regulatory bodies tasked with maintaining fair and competitive markets. The study provides a foundation for further exploration into the ramifications of deferring high-stakes decisions to LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.13054v1">LLM Surgery: Efficient Knowledge Unlearning and Editing in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-09-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized various domains, yet their utility comes with significant challenges related to outdated or problematic knowledge embedded during pretraining. This paper addresses the challenge of modifying LLMs to unlearn problematic and outdated information while efficiently integrating new knowledge without retraining from scratch. Here, we propose LLM Surgery, a framework to efficiently modify LLM behaviour by optimizing a three component objective function that: (1) Performs reverse gradient on unlearning dataset (problematic and outdated information), (2) Performs gradient descent on the update dataset (new and updated information), and (3) Minimizes the KL divergence on the retain dataset (small subset of unchanged text), ensuring alignment between pretrained and modified model outputs. Due to the lack of publicly available datasets specifically tailored for our novel task, we compiled a new dataset and an evaluation benchmark. Using Llama2-7B, we demonstrate that LLM Surgery can achieve significant forgetting on the unlearn set, a 20\% increase in accuracy on the update set, and maintain performance on the retain set.
    </details>
</div>
