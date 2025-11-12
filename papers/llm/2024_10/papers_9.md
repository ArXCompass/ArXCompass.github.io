# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- Part 9
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16700v2">Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 NeurIPS 2024. Code: https://github.com/mshukor/ima-lmms. Project page: https://ima-lmms.github.io/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance on multimodal tasks, without any multimodal finetuning. They are the building block for Large Multimodal Models, yet, we still lack a proper understanding of their success. In this work, we expose frozen LLMs to image, video, audio and text inputs and analyse their internal representation aiming to understand their generalization beyond textual inputs. Findings. Perceptual tokens (1) are easily distinguishable from textual ones inside LLMs, with significantly different representations, and complete translation to textual tokens does not exist. Yet, (2) both perceptual and textual tokens activate similar LLM weights. Despite being different, (3) perceptual and textual tokens are implicitly aligned inside LLMs, we call this the implicit multimodal alignment (IMA), and argue that this is linked to architectural design, helping LLMs to generalize. This provide more evidence to believe that the generalization of LLMs to multimodal inputs is mainly due to their architecture. Implications. (1) We find a positive correlation between the implicit alignment score and the task performance, suggesting that this could act as a proxy metric for model evaluation and selection. (2) A negative correlation exists regarding hallucinations, revealing that this problem is mainly due to misalignment between the internal perceptual and textual representations. (3) Perceptual tokens change slightly throughout the model, thus, we propose different approaches to skip computations (e.g. in FFN layers), and significantly reduce the inference cost. (4) Due to the slowly changing embeddings across layers, and the high overlap between textual and multimodal activated weights, we compress LLMs by keeping only 1 subnetwork that works well across a wide range of multimodal tasks. Paper code: https://github.com/mshukor/ima-lmms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.10019v3">R-Judge: Benchmarking Safety Risk Awareness for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on the harmlessness of LLM-generated content in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging and identifying safety risks given agent interaction records. R-Judge comprises 569 records of multi-turn agent interaction, encompassing 27 key risk scenarios among 5 application categories and 10 risk types. It is of high-quality curation with annotated safety labels and risk descriptions. Evaluation of 11 LLMs on R-Judge shows considerable room for enhancing the risk awareness of LLMs: The best-performing model, GPT-4o, achieves 74.42% while no other models significantly exceed the random. Moreover, we reveal that risk awareness in open agent scenarios is a multi-dimensional capability involving knowledge and reasoning, thus challenging for LLMs. With further experiments, we find that fine-tuning on safety judgment significantly improve model performance while straightforward prompting mechanisms fail. R-Judge is publicly available at https://github.com/Lordog/R-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18892v2">IDGen: Item Discrimination Induced Prompt Generation for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow increasingly adept at managing complex tasks, the evaluation set must keep pace with these advancements to ensure it remains sufficiently discriminative. Item Discrimination (ID) theory, which is widely used in educational assessment, measures the ability of individual test items to differentiate between high and low performers. Inspired by this theory, we propose an ID-induced prompt synthesis framework for evaluating LLMs to ensure the evaluation set can continually update and refine according to model abilities. Our data synthesis framework prioritizes both breadth and specificity. It can generate prompts that comprehensively evaluate the capabilities of LLMs while revealing meaningful performance differences between models, allowing for effective discrimination of their relative strengths and weaknesses across various tasks and domains. To produce high-quality data, we incorporate a self-correct mechanism into our generalization framework, and develop two models to predict prompt discrimination and difficulty score to facilitate our data synthesis framework, contributing valuable tools to evaluation data synthesis research. We apply our generated data to evaluate five SOTA models. Our data achieves an average score of 51.92, accompanied by a variance of 10.06. By contrast, previous works (i.e., SELF-INSTRUCT and WizardLM) obtain an average score exceeding 67, with a variance below 3.2. The results demonstrate that the data generated by our framework is more challenging and discriminative compared to previous works. We will release a dataset of over 3,000 carefully crafted prompts to facilitate evaluation research of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05318v1">Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Despite significant advancements in the general capability of large language models (LLMs), they continue to struggle with consistent and accurate reasoning, especially in complex tasks such as mathematical and code reasoning. One key limitation is that LLMs are trained primarily on correct solutions, reducing their ability to detect and learn from errors, which hampers their ability to reliably verify and rank outputs. To address this, we scale up the inference-time computation by generating multiple reasoning paths and employing verifiers to assess and rank the generated outputs by correctness. To facilitate this, we introduce a comprehensive dataset consisting of correct and incorrect solutions for math and code tasks, generated by multiple LLMs. This diverse set of solutions enables verifiers to more effectively distinguish and rank correct answers from erroneous outputs. The training methods for building verifiers were selected based on an extensive comparison of existing approaches. Moreover, to leverage the unique strengths of different reasoning strategies, we propose a novel collaborative method integrating Chain-of-Thought (CoT) and Program-of-Thought (PoT) solutions for verification. CoT provides a clear, step-by-step reasoning process that enhances interpretability, while PoT, being executable, offers a precise and error-sensitive validation mechanism. By taking both of their strengths, our approach significantly improves the accuracy and reliability of reasoning verification. Our verifiers, Math-Rev and Code-Rev, demonstrate substantial performance gains to existing LLMs, achieving state-of-the-art results on benchmarks such as GSM8k and MATH and even outperforming GPT-4o with Qwen-72B-Instruct as the reasoner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20774v2">Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 31 pages, including main paper, references, and appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04035v1">Gamifying XAI: Enhancing AI Explainability for Non-technical Users through LLM-Powered Narrative Gamifications</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) has become tightly integrated into modern technology, yet existing exploratory visualizations for explainable AI (XAI) are primarily designed for users with technical expertise. This leaves everyday users, who also regularly interact with AI systems, with limited resources to explore or understand AI technologies they use. We propose a novel framework that enables non-technical users to collect insights by conversing directly with visualization elements via LLM-powered narrative gamifications. We implemented a prototype that utilizes such gamification to facilitate non-technical users' exploration of AI embedding projections. We conducted a comparative study with 10 participants to assess our prototype quantitatively and qualitatively. Our study results indicate that although our prototype effectively enhances non-technical users' AI/XAI knowledge, and users believe they learn more through the gamification feature, it remains inconclusive whether the gamification itself leads to further improvements in understanding. In addition, opinions among participants regarding the framework's engagement are mixed: some believe it enhances their exploration of the visualizations, while others feel it disrupts their workflow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01171v3">Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 Accepted to EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      The concept of persona, originally adopted in dialogue literature, has re-surged as a promising framework for tailoring large language models (LLMs) to specific context (e.g., personalized search, LLM-as-a-judge). However, the growing research on leveraging persona in LLMs is relatively disorganized and lacks a systematic taxonomy. To close the gap, we present a comprehensive survey to categorize the current state of the field. We identify two lines of research, namely (1) LLM Role-Playing, where personas are assigned to LLMs, and (2) LLM Personalization, where LLMs take care of user personas. Additionally, we introduce existing methods for LLM personality evaluation. To the best of our knowledge, we present the first survey for role-playing and personalization in LLMs under the unified view of persona. We continuously maintain a paper collection to foster future endeavors: https://github.com/MiuLab/PersonaLLM-Survey
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04009v1">ASPIRER: Bypassing System Prompts With Permutation-based Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to many applications, with system prompts serving as a key mechanism to regulate model behavior and ensure ethical outputs. In this paper, we introduce a novel backdoor attack that systematically bypasses these system prompts, posing significant risks to the AI supply chain. Under normal conditions, the model adheres strictly to its system prompts. However, our backdoor allows malicious actors to circumvent these safeguards when triggered. Specifically, we explore a scenario where an LLM provider embeds a covert trigger within the base model. A downstream deployer, unaware of the hidden trigger, fine-tunes the model and offers it as a service to users. Malicious actors can purchase the trigger from the provider and use it to exploit the deployed model, disabling system prompts and achieving restricted outcomes. Our attack utilizes a permutation trigger, which activates only when its components are arranged in a precise order, making it computationally challenging to detect or reverse-engineer. We evaluate our approach on five state-of-the-art models, demonstrating that our method achieves an attack success rate (ASR) of up to 99.50% while maintaining a clean accuracy (CACC) of 98.58%, even after defensive fine-tuning. These findings highlight critical vulnerabilities in LLM deployment pipelines and underscore the need for stronger defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17302v3">Can LLM Generate Culturally Relevant Commonsense QA Data? Case Study in Indonesian and Sundanese</a></div>
    <div class="paper-meta">
      📅 2024-10-05
      | 💬 EMNLP 2024 Camera-Ready
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used to generate synthetic data for training and evaluating models. However, it is unclear whether they can generate a good quality of question answering (QA) dataset that incorporates knowledge and cultural nuance embedded in a language, especially for low-resource languages. In this study, we investigate the effectiveness of using LLMs in generating culturally relevant commonsense QA datasets for Indonesian and Sundanese languages. To do so, we create datasets for these languages using various methods involving both LLMs and human annotators, resulting in ~4.5K questions per language (~9K in total), making our dataset the largest of its kind. Our experiments show that automatic data adaptation from an existing English dataset is less effective for Sundanese. Interestingly, using the direct generation method on the target language, GPT-4 Turbo can generate questions with adequate general knowledge in both languages, albeit not as culturally 'deep' as humans. We also observe a higher occurrence of fluency errors in the Sundanese dataset, highlighting the discrepancy between medium- and lower-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03997v1">YOLO-MARL: You Only LLM Once for Multi-agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-05
    </div>
    <details class="paper-abstract">
      Advancements in deep multi-agent reinforcement learning (MARL) have positioned it as a promising approach for decision-making in cooperative games. However, it still remains challenging for MARL agents to learn cooperative strategies for some game environments. Recently, large language models (LLMs) have demonstrated emergent reasoning capabilities, making them promising candidates for enhancing coordination among the agents. However, due to the model size of LLMs, it can be expensive to frequently infer LLMs for actions that agents can take. In this work, we propose You Only LLM Once for MARL (YOLO-MARL), a novel framework that leverages the high-level task planning capabilities of LLMs to improve the policy learning process of multi-agents in cooperative games. Notably, for each game environment, YOLO-MARL only requires one time interaction with LLMs in the proposed strategy generation, state interpretation and planning function generation modules, before the MARL policy training process. This avoids the ongoing costs and computational time associated with frequent LLMs API calls during training. Moreover, the trained decentralized normal-sized neural network-based policies operate independently of the LLM. We evaluate our method across three different environments and demonstrate that YOLO-MARL outperforms traditional MARL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16732v2">"It Explains What I am Currently Going Through Perfectly to a Tee": Understanding User Perceptions on LLM-Enhanced Narrative Interventions</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Stories about overcoming personal struggles can effectively illustrate the application of psychological theories in real life, yet they may fail to resonate with individuals' experiences. In this work, we employ large language models (LLMs) to create tailored narratives that acknowledge and address unique challenging thoughts and situations faced by individuals. Our study, involving 346 young adults across two settings, demonstrates that LLM-enhanced stories were perceived to be better than human-written ones in conveying key takeaways, promoting reflection, and reducing belief in negative thoughts. These stories were not only seen as more relatable but also similarly authentic to human-written ones, highlighting the potential of LLMs in helping young adults manage their struggles. The findings of this work provide crucial design considerations for future narrative-based digital mental health interventions, such as the need to maintain relatability without veering into implausibility and refining the wording and tone of AI-enhanced content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03953v1">LLM-TOPLA: Efficient LLM Ensemble by Maximising Diversity</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Combining large language models during training or at inference time has shown substantial performance gain over component LLMs. This paper presents LLM-TOPLA, a diversity-optimized LLM ensemble method with three unique properties: (i) We introduce the focal diversity metric to capture the diversity-performance correlation among component LLMs of an ensemble. (ii) We develop a diversity-optimized ensemble pruning algorithm to select the top-k sub-ensembles from a pool of $N$ base LLMs. Our pruning method recommends top-performing LLM subensembles of size $S$, often much smaller than $N$. (iii) We generate new output for each prompt query by utilizing a learn-to-ensemble approach, which learns to detect and resolve the output inconsistency among all component LLMs of an ensemble. Extensive evaluation on four different benchmarks shows good performance gain over the best LLM ensemble methods: (i) In constrained solution set problems, LLM-TOPLA outperforms the best-performing ensemble (Mixtral) by 2.2\% in accuracy on MMLU and the best-performing LLM ensemble (MoreAgent) on GSM8k by 2.1\%. (ii) In generative tasks, LLM-TOPLA outperforms the top-2 performers (Llama70b/Mixtral) on SearchQA by $3.9\mathrm{x}$ in F1, and on XSum by more than $38$ in ROUGE-1. Our code and dataset, which contains outputs of 8 modern LLMs on 4 benchmarks is available at https://github.com/git-disl/llm-topla
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14673v2">Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit positional bias, struggling to utilize information from the middle or end of long contexts. Our study explores LLMs' long-context reasoning by probing their hidden representations. We find that while LLMs encode the position of target information, they often fail to leverage this in generating accurate responses. This reveals a disconnect between information retrieval and utilization, a "know but don't tell" phenomenon. We further analyze the relationship between extraction time and final accuracy, offering insights into the underlying mechanics of transformer models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20365v2">VideoINSTA: Zero-shot Long Video Understanding via Informative Spatial-Temporal Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP 2024 Findings; 22 pages; Code: https://github.com/mayhugotong/VideoINSTA
    </div>
    <details class="paper-abstract">
      In the video-language domain, recent works in leveraging zero-shot Large Language Model-based reasoning for video understanding have become competitive challengers to previous end-to-end models. However, long video understanding presents unique challenges due to the complexity of reasoning over extended timespans, even for zero-shot LLM-based approaches. The challenge of information redundancy in long videos prompts the question of what specific information is essential for large language models (LLMs) and how to leverage them for complex spatial-temporal reasoning in long-form video analysis. We propose a framework VideoINSTA, i.e. INformative Spatial-TemporAl Reasoning for zero-shot long-form video understanding. VideoINSTA contributes (1) a zero-shot framework for long video understanding using LLMs; (2) an event-based temporal reasoning and content-based spatial reasoning approach for LLMs to reason over spatial-temporal information in videos; (3) a self-reflective information reasoning scheme balancing temporal factors based on information sufficiency and prediction confidence. Our model significantly improves the state-of-the-art on three long video question-answering benchmarks: EgoSchema, NextQA, and IntentQA, and the open question answering dataset ActivityNetQA. The code is released here: https://github.com/mayhugotong/VideoINSTA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09080v1">Leveraging Social Determinants of Health in Alzheimer's Research Using LLM-Augmented Literature Mining and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Growing evidence suggests that social determinants of health (SDoH), a set of nonmedical factors, affect individuals' risks of developing Alzheimer's disease (AD) and related dementias. Nevertheless, the etiological mechanisms underlying such relationships remain largely unclear, mainly due to difficulties in collecting relevant information. This study presents a novel, automated framework that leverages recent advancements of large language model (LLM) and natural language processing techniques to mine SDoH knowledge from extensive literature and integrate it with AD-related biological entities extracted from the general-purpose knowledge graph PrimeKG. Utilizing graph neural networks, we performed link prediction tasks to evaluate the resultant SDoH-augmented knowledge graph. Our framework shows promise for enhancing knowledge discovery in AD and can be generalized to other SDoH-related research areas, offering a new tool for exploring the impact of social determinants on health outcomes. Our code is available at: https://github.com/hwq0726/SDoHenPKG
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03656v6">Emotionally Numb or Empathetic? Evaluating How LLMs Feel Using EmotionBench</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 NeurIPS 2024; 10 pages of main text; 14 pages of appendices
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models' (LLMs) anthropomorphic capabilities has become increasingly important in contemporary discourse. Utilizing the emotion appraisal theory from psychology, we propose to evaluate the empathy ability of LLMs, i.e., how their feelings change when presented with specific situations. After a careful and comprehensive survey, we collect a dataset containing over 400 situations that have proven effective in eliciting the eight emotions central to our study. Categorizing the situations into 36 factors, we conduct a human evaluation involving more than 1,200 subjects worldwide. With the human evaluation results as references, our evaluation includes seven LLMs, covering both commercial and open-source models, including variations in model sizes, featuring the latest iterations, such as GPT-4, Mixtral-8x22B, and LLaMA-3.1. We find that, despite several misalignments, LLMs can generally respond appropriately to certain situations. Nevertheless, they fall short in alignment with the emotional behaviors of human beings and cannot establish connections between similar situations. Our collected dataset of situations, the human evaluation results, and the code of our testing framework, i.e., EmotionBench, are publicly available at https://github.com/CUHK-ARISE/EmotionBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.12060v2">Evidence-backed Fact Checking using RAG and Few-Shot In-Context Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted in The Seventh FEVER Workshop at EMNLP 2024
    </div>
    <details class="paper-abstract">
      Given the widespread dissemination of misinformation on social media, implementing fact-checking mechanisms for online claims is essential. Manually verifying every claim is very challenging, underscoring the need for an automated fact-checking system. This paper presents our system designed to address this issue. We utilize the Averitec dataset (Schlichtkrull et al., 2023) to assess the performance of our fact-checking system. In addition to veracity prediction, our system provides supporting evidence, which is extracted from the dataset. We develop a Retrieve and Generate (RAG) pipeline to extract relevant evidence sentences from a knowledge base, which are then inputted along with the claim into a large language model (LLM) for classification. We also evaluate the few-shot In-Context Learning (ICL) capabilities of multiple LLMs. Our system achieves an 'Averitec' score of 0.33, which is a 22% absolute improvement over the baseline. Our Code is publicly available on https://github.com/ronit-singhal/evidence-backed-fact-checking-using-rag-and-few-shot-in-context-learning-with-llms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03864v1">DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05306v1">Towards Assuring EU AI Act Compliance and Adversarial Robustness of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted in the AI Act Workshop
    </div>
    <details class="paper-abstract">
      Large language models are prone to misuse and vulnerable to security threats, raising significant safety and security concerns. The European Union's Artificial Intelligence Act seeks to enforce AI robustness in certain contexts, but faces implementation challenges due to the lack of standards, complexity of LLMs and emerging security vulnerabilities. Our research introduces a framework using ontologies, assurance cases, and factsheets to support engineers and stakeholders in understanding and documenting AI system compliance and security regarding adversarial robustness. This approach aims to ensure that LLMs adhere to regulatory standards and are equipped to counter potential threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09078v1">Knowledge-Augmented Reasoning for EUAIA Compliance and Adversarial Robustness of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted in the VECOMP 2024 workshop
    </div>
    <details class="paper-abstract">
      The EU AI Act (EUAIA) introduces requirements for AI systems which intersect with the processes required to establish adversarial robustness. However, given the ambiguous language of regulation and the dynamicity of adversarial attacks, developers of systems with highly complex models such as LLMs may find their effort to be duplicated without the assurance of having achieved either compliance or robustness. This paper presents a functional architecture that focuses on bridging the two properties, by introducing components with clear reference to their source. Taking the detection layer recommended by the literature, and the reporting layer required by the law, we aim to support developers and auditors with a reasoning layer based on knowledge augmentation (rules, assurance cases, contextual mappings). Our findings demonstrate a novel direction for ensuring LLMs deployed in the EU are both compliant and adversarially robust, which underpin trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05304v1">Developing Assurance Cases for Adversarial Robustness and Regulatory Compliance in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted to the ASSURE 2024 workshop
    </div>
    <details class="paper-abstract">
      This paper presents an approach to developing assurance cases for adversarial robustness and regulatory compliance in large language models (LLMs). Focusing on both natural and code language tasks, we explore the vulnerabilities these models face, including adversarial attacks based on jailbreaking, heuristics, and randomization. We propose a layered framework incorporating guardrails at various stages of LLM deployment, aimed at mitigating these attacks and ensuring compliance with the EU AI Act. Our approach includes a meta-layer for dynamic risk management and reasoning, crucial for addressing the evolving nature of LLM vulnerabilities. We illustrate our method with two exemplary assurance cases, highlighting how different contexts demand tailored strategies to ensure robust and compliant AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.04883v4">Multi-User Chat Assistant (MUCA): a Framework Using LLMs to Facilitate Group Conversations</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have provided a new avenue for chatbot development. Most existing research, however, has primarily centered on single-user chatbots that determine "What" to answer. This paper highlights the complexity of multi-user chatbots, introducing the 3W design dimensions: "What" to say, "When" to respond, and "Who" to answer. Additionally, we proposed Multi-User Chat Assistant (MUCA), an LLM-based framework tailored for group discussions. MUCA consists of three main modules: Sub-topic Generator, Dialog Analyzer, and Conversational Strategies Arbitrator. These modules jointly determine suitable response contents, timings, and appropriate addressees. This paper further proposes an LLM-based Multi-User Simulator (MUS) to ease MUCA's optimization, enabling faster simulation of conversations between the chatbot and simulated users, and speeding up MUCA's early development. In goal-oriented conversations with a small to medium number of participants, MUCA demonstrates effectiveness in tasks like chiming in at appropriate timings, generating relevant content, and improving user engagement, as shown by case studies and user studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.16310v3">An Insight into Security Code Review with LLMs: Capabilities, Obstacles and Influential Factors</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 26 pages, 5 images, 7 tables, Manuscript submitted to a journal (2024)
    </div>
    <details class="paper-abstract">
      Security code review is a time-consuming and labor-intensive process typically requiring integration with automated security defect detection tools. However, existing security analysis tools struggle with poor generalization, high false positive rates, and coarse detection granularity. Large Language Models (LLMs) have been considered promising candidates for addressing those challenges. In this study, we conducted an empirical study to explore the potential of LLMs in detecting security defects during code review. Specifically, we evaluated the performance of six LLMs under five different prompts and compared them with state-of-theart static analysis tools. We also performed linguistic and regression analyses for the best-performing LLM to identify quality problems in its responses and factors influencing its performance. Our findings show that: (1) existing pre-trained LLMs have limited capability in security code review but? significantly outperform the state-of-the-art static analysis tools. (2) GPT-4 performs best among all LLMs when provided with a CWE list for reference. (3) GPT-4 frequently generates responses that are verbose or not compliant with the task requirements given in the prompts. (4) GPT-4 is more adept at identifying security defects in code files with fewer tokens, containing functional logic, or written by developers with less involvement in the project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03834v1">GraphRouter: A Graph-based Router for LLM Selections</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      The rapidly growing number and variety of Large Language Models (LLMs) present significant challenges in efficiently selecting the appropriate LLM for a given query, especially considering the trade-offs between performance and computational cost. Current LLM selection methods often struggle to generalize across new LLMs and different tasks because of their limited ability to leverage contextual interactions among tasks, queries, and LLMs, as well as their dependence on a transductive learning framework. To address these shortcomings, we introduce a novel inductive graph framework, named as GraphRouter, which fully utilizes the contextual information among tasks, queries, and LLMs to enhance the LLM selection process. GraphRouter constructs a heterogeneous graph comprising task, query, and LLM nodes, with interactions represented as edges, which efficiently captures the contextual information between the query's requirements and the LLM's capabilities. Through an innovative edge prediction mechanism, GraphRouter is able to predict attributes (the effect and cost of LLM response) of potential edges, allowing for optimized recommendations that adapt to both existing and newly introduced LLMs without requiring retraining. Comprehensive experiments across three distinct effect-cost weight scenarios have shown that GraphRouter substantially surpasses existing routers, delivering a minimum performance improvement of 12.3%. In addition, it achieves enhanced generalization across new LLMs settings and supports diverse tasks with at least a 9.5% boost in effect and a significant reduction in computational demands. This work endeavors to apply a graph-based approach for the contextual and adaptive selection of LLMs, offering insights for real-world applications. Our codes for GraphRouter will soon be released at https://github.com/ulab-uiuc/GraphRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03645v1">GenSim2: Scaling Robot Data Generation with Multi-modal and Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 CoRL 2024. Project website: https://gensim2.github.io/
    </div>
    <details class="paper-abstract">
      Robotic simulation today remains challenging to scale up due to the human efforts required to create diverse simulation tasks and scenes. Simulation-trained policies also face scalability issues as many sim-to-real methods focus on a single task. To address these challenges, this work proposes GenSim2, a scalable framework that leverages coding LLMs with multi-modal and reasoning capabilities for complex and realistic simulation task creation, including long-horizon tasks with articulated objects. To automatically generate demonstration data for these tasks at scale, we propose planning and RL solvers that generalize within object categories. The pipeline can generate data for up to 100 articulated tasks with 200 objects and reduce the required human efforts. To utilize such data, we propose an effective multi-task language-conditioned policy architecture, dubbed proprioceptive point-cloud transformer (PPT), that learns from the generated demonstrations and exhibits strong sim-to-real zero-shot transfer. Combining the proposed pipeline and the policy architecture, we show a promising usage of GenSim2 that the generated data can be used for zero-shot transfer or co-train with real-world collected data, which enhances the policy performance by 20% compared with training exclusively on limited real data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06520v2">Retrieval-Augmented Hierarchical in-Context Reinforcement Learning and Hindsight Modular Reflections for Task Planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities in various language tasks, making them promising candidates for decision-making in robotics. Inspired by Hierarchical Reinforcement Learning (HRL), we propose Retrieval-Augmented in-context reinforcement Learning (RAHL), a novel framework that decomposes complex tasks into sub-tasks using an LLM-based high-level policy, in which a complex task is decomposed into sub-tasks by a high-level policy on-the-fly. The sub-tasks, defined by goals, are assigned to the low-level policy to complete. To improve the agent's performance in multi-episode execution, we propose Hindsight Modular Reflection (HMR), where, instead of reflecting on the full trajectory, we let the agent reflect on shorter sub-trajectories to improve reflection efficiency. We evaluated the decision-making ability of the proposed RAHL in three benchmark environments--ALFWorld, Webshop, and HotpotQA. The results show that RAHL can achieve an improvement in performance in 9%, 42%, and 10% in 5 episodes of execution in strong baselines. Furthermore, we also implemented RAHL on the Boston Dynamics SPOT robot. The experiment shows that the robot can scan the environment, find entrances, and navigate to new rooms controlled by the LLM policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20974v3">SaySelf: Teaching LLMs to Express Confidence with Self-Reflective Rationales</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate inaccurate or fabricated information and generally fail to indicate their confidence, which limits their broader applications. Previous work elicits confidence from LLMs by direct or self-consistency prompting, or constructing specific datasets for supervised finetuning. The prompting-based approaches have inferior performance, and the training-based approaches are limited to binary or inaccurate group-level confidence estimates. In this work, we present the advanced SaySelf, a training framework that teaches LLMs to express more accurate fine-grained confidence estimates. In addition, beyond the confidence scores, SaySelf initiates the process of directing LLMs to produce self-reflective rationales that clearly identify gaps in their parametric knowledge and explain their uncertainty. This is achieved by using an LLM to automatically summarize the uncertainties in specific knowledge via natural language. The summarization is based on the analysis of the inconsistency in multiple sampled reasoning chains, and the resulting data is utilized for supervised fine-tuning. Moreover, we utilize reinforcement learning with a meticulously crafted reward function to calibrate the confidence estimates, motivating LLMs to deliver accurate, high-confidence predictions and to penalize overconfidence in erroneous outputs. Experimental results in both in-distribution and out-of-distribution datasets demonstrate the effectiveness of SaySelf in reducing the confidence calibration error and maintaining the task performance. We show that the generated self-reflective rationales are reasonable and can further contribute to the calibration. The code is made public at https://github.com/xu1868/SaySelf.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03608v1">TICKing All the Boxes: Generated Checklists Improve LLM Evaluation and Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Given the widespread adoption and usage of Large Language Models (LLMs), it is crucial to have flexible and interpretable evaluations of their instruction-following ability. Preference judgments between model outputs have become the de facto evaluation standard, despite distilling complex, multi-faceted preferences into a single ranking. Furthermore, as human annotation is slow and costly, LLMs are increasingly used to make these judgments, at the expense of reliability and interpretability. In this work, we propose TICK (Targeted Instruct-evaluation with ChecKlists), a fully automated, interpretable evaluation protocol that structures evaluations with LLM-generated, instruction-specific checklists. We first show that, given an instruction, LLMs can reliably produce high-quality, tailored evaluation checklists that decompose the instruction into a series of YES/NO questions. Each question asks whether a candidate response meets a specific requirement of the instruction. We demonstrate that using TICK leads to a significant increase (46.4% $\to$ 52.2%) in the frequency of exact agreements between LLM judgements and human preferences, as compared to having an LLM directly score an output. We then show that STICK (Self-TICK) can be used to improve generation quality across multiple benchmarks via self-refinement and Best-of-N selection. STICK self-refinement on LiveBench reasoning tasks leads to an absolute gain of $+$7.8%, whilst Best-of-N selection with STICK attains $+$6.3% absolute improvement on the real-world instruction dataset, WildBench. In light of this, structured, multi-faceted self-improvement is shown to be a promising way to further advance LLM capabilities. Finally, by providing LLM-generated checklists to human evaluators tasked with directly scoring LLM responses to WildBench instructions, we notably increase inter-annotator agreement (0.194 $\to$ 0.256).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13213v2">Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      We study 14 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigororous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03568v1">Towards Linguistically-Aware and Language-Independent Tokenization for Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive study on the tokenization techniques employed by state-of-the-art large language models (LLMs) and their implications on the cost and availability of services across different languages, especially low resource languages. The analysis considers multiple LLMs, including GPT-4 (using cl100k_base embeddings), GPT-3 (with p50k_base embeddings), and DaVinci (employing r50k_base embeddings), as well as the widely used BERT base tokenizer. The study evaluates the tokenization variability observed across these models and investigates the challenges of linguistic representation in subword tokenization. The research underscores the importance of fostering linguistically-aware development practices, especially for languages that are traditionally under-resourced. Moreover, this paper introduces case studies that highlight the real-world implications of tokenization choices, particularly in the context of electronic health record (EHR) systems. This research aims to promote generalizable Internationalization (I18N) practices in the development of AI services in this domain and beyond, with a strong emphasis on inclusivity, particularly for languages traditionally underrepresented in AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12821v3">Identifying Factual Inconsistencies in Summaries: Grounding LLM Inference via Task Taxonomy</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted to EMNLP 2024 Findings
    </div>
    <details class="paper-abstract">
      Factual inconsistencies pose a significant hurdle for the faithful summarization by generative models. While a major direction to enhance inconsistency detection is to derive stronger Natural Language Inference (NLI) models, we propose an orthogonal aspect that underscores the importance of incorporating task-specific taxonomy into the inference. To this end, we consolidate key error types of inconsistent facts in summaries, and incorporate them to facilitate both the zero-shot and supervised paradigms of LLMs. Extensive experiments on ten datasets of five distinct domains suggest that, zero-shot LLM inference could benefit from the explicit solution space depicted by the error type taxonomy, and achieves state-of-the-art performance overall, surpassing specialized non-LLM baselines, as well as recent LLM baselines. We further distill models that fuse the taxonomy into parameters through our designed prompt completions and supervised training strategies, efficiently substituting state-of-the-art zero-shot inference with much larger LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03537v1">Ward: Provable RAG Dataset Inference via LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) improves LLMs by enabling them to incorporate external data during generation. This raises concerns for data owners regarding unauthorized use of their content in RAG systems. Despite its importance, the challenge of detecting such unauthorized usage remains underexplored, with existing datasets and methodologies from adjacent fields being ill-suited for its study. In this work, we take several steps to bridge this gap. First, we formalize this problem as (black-box) RAG Dataset Inference (RAG-DI). To facilitate research on this challenge, we further introduce a novel dataset specifically designed for benchmarking RAG-DI methods under realistic conditions, and propose a set of baseline approaches. Building on this foundation, we introduce Ward, a RAG-DI method based on LLM watermarks that enables data owners to obtain rigorous statistical guarantees regarding the usage of their dataset in a RAG system. In our experimental evaluation, we show that Ward consistently outperforms all baselines across many challenging settings, achieving higher accuracy, superior query efficiency and robustness. Our work provides a foundation for future studies of RAG-DI and highlights LLM watermarks as a promising approach to this problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17600v2">"Seeing the Big through the Small": Can LLMs Approximate Human Judgment Distributions on NLI from a Few Explanations?</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted by EMNLP 2024 Findings, 24 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Human label variation (HLV) is a valuable source of information that arises when multiple human annotators provide different labels for valid reasons. In Natural Language Inference (NLI) earlier approaches to capturing HLV involve either collecting annotations from many crowd workers to represent human judgment distribution (HJD) or use expert linguists to provide detailed explanations for their chosen labels. While the former method provides denser HJD information, obtaining it is resource-intensive. In contrast, the latter offers richer textual information but it is challenging to scale up to many human judges. Besides, large language models (LLMs) are increasingly used as evaluators ("LLM judges") but with mixed results, and few works aim to study HJDs. This study proposes to exploit LLMs to approximate HJDs using a small number of expert labels and explanations. Our experiments show that a few explanations significantly improve LLMs' ability to approximate HJDs with and without explicit labels, thereby providing a solution to scale up annotations for HJD. However, fine-tuning smaller soft-label aware models with the LLM-generated model judgment distributions (MJDs) presents partially inconsistent results: while similar in distance, their resulting fine-tuned models and visualized distributions differ substantially. We show the importance of complementing instance-level distance measures with a global-level shape metric and visualization to more effectively evaluate MJDs against human judgment distributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15352v2">A SMART Mnemonic Sounds like "Glue Tonic": Mixing LLMs with Student Feedback to Make Mnemonic Learning Stick</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Keyword mnemonics are memorable explanations that link new terms to simpler keywords. Prior work generates mnemonics for students, but they do not train models using mnemonics students prefer and aid learning. We build SMART, a mnemonic generator trained on feedback from real students learning new terms. To train SMART, we first fine-tune LLaMA-2 on a curated set of user-written mnemonics. We then use LLM alignment to enhance SMART: we deploy mnemonics generated by SMART in a flashcard app to find preferences on mnemonics students favor. We gather 2684 preferences from 45 students across two types: expressed (inferred from ratings) and observed (inferred from student learning), yielding three key findings. First, expressed and observed preferences disagree; what students think is helpful does not always capture what is truly helpful. Second, Bayesian models can synthesize complementary data from multiple preference types into a single effectiveness signal. SMART is tuned via Direct Preference Optimization on this signal, which resolves ties and missing labels in the typical method of pairwise comparisons, augmenting data for LLM output quality gains. Third, mnemonic experts assess SMART as matching GPT-4 at much lower deployment costs, showing the utility of capturing diverse student feedback to align LLMs in education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05673v3">Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample diverse reasoning paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across five challenging puzzle-solving tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), and PrOntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03492v1">Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 4 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are stochastic, and not all models give deterministic answers, even when setting temperature to zero with a fixed random seed. However, few benchmark studies attempt to quantify uncertainty, partly due to the time and cost of repeated experiments. We use benchmarks designed for testing LLMs' capacity to reason about cardinal directions to explore the impact of experimental repeats on mean score and prediction interval. We suggest a simple method for cost-effectively quantifying the uncertainty of a benchmark score and make recommendations concerning reproducible LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02338v2">How Much Can RAG Help the Reasoning of LLM?</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has gained significant popularity in modern Large Language Models (LLMs) due to its effectiveness in introducing new knowledge and reducing hallucinations. However, the deep understanding of RAG remains limited, how does RAG help the reasoning process and can RAG help improve the reasoning capability remains question. While external documents are typically considered as a method to incorporate domain-specific information, they also contain intermediate reasoning results related to the query, this suggests that documents could enhance the reasoning capability of LLMs, which has not been previously explored. In this paper, we investigate this issue in depth and find that while RAG can assist with reasoning, the help is limited. If we conceptualize the reasoning process as a tree with fixed depth, then RAG struggles to assist LLMs in performing deeper reasoning. Additionally, the information in the documents requires preprocessing to filter out noise. We demonstrate that this preprocessing is difficult to achieve simply fine-tuning of the LLM, it often necessitates numerous additional transformer layers to solve the problem. To simplify the problem, we propose DPrompt tuning, which effectively resolves the issue within just limited transformer layers, leading to improved performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.10902v3">Soda-Eval: Open-Domain Dialogue Evaluation in the age of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted to EMNLP2024 (findings)
    </div>
    <details class="paper-abstract">
      Although human evaluation remains the gold standard for open-domain dialogue evaluation, the growing popularity of automated evaluation using Large Language Models (LLMs) has also extended to dialogue. However, most frameworks leverage benchmarks that assess older chatbots on aspects such as fluency and relevance, which are not reflective of the challenges associated with contemporary models. In fact, a qualitative analysis on Soda, a GPT-3.5 generated dialogue dataset, suggests that current chatbots may exhibit several recurring issues related to coherence and commonsense knowledge, but generally produce highly fluent and relevant responses. Noting the aforementioned limitations, this paper introduces Soda-Eval, an annotated dataset based on Soda that covers over 120K turn-level assessments across 10K dialogues, where the annotations were generated by GPT-4. Using Soda-Eval as a benchmark, we then study the performance of several open-access instruction-tuned LLMs, finding that dialogue evaluation remains challenging. Fine-tuning these models improves performance over few-shot inferences, both in terms of correlation and explanation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03466v1">Is Safer Better? The Impact of Guardrails on the Argumentative Strength of LLMs in Hate Speech Countering</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 To appear in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (long paper)
    </div>
    <details class="paper-abstract">
      The potential effectiveness of counterspeech as a hate speech mitigation strategy is attracting increasing interest in the NLG research community, particularly towards the task of automatically producing it. However, automatically generated responses often lack the argumentative richness which characterises expert-produced counterspeech. In this work, we focus on two aspects of counterspeech generation to produce more cogent responses. First, by investigating the tension between helpfulness and harmlessness of LLMs, we test whether the presence of safety guardrails hinders the quality of the generations. Secondly, we assess whether attacking a specific component of the hate speech results in a more effective argumentative strategy to fight online hate. By conducting an extensive human and automatic evaluation, we show how the presence of safety guardrails can be detrimental also to a task that inherently aims at fostering positive social interactions. Moreover, our results show that attacking a specific component of the hate speech, and in particular its implicit negative stereotype and its hateful parts, leads to higher-quality generations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03457v1">CoCoLoFa: A Dataset of News Comments with Common Logical Fallacies Written by LLM-Assisted Crowds</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP 2024)
    </div>
    <details class="paper-abstract">
      Detecting logical fallacies in texts can help users spot argument flaws, but automating this detection is not easy. Manually annotating fallacies in large-scale, real-world text data to create datasets for developing and validating detection models is costly. This paper introduces CoCoLoFa, the largest known logical fallacy dataset, containing 7,706 comments for 648 news articles, with each comment labeled for fallacy presence and type. We recruited 143 crowd workers to write comments embodying specific fallacy types (e.g., slippery slope) in response to news articles. Recognizing the complexity of this writing task, we built an LLM-powered assistant into the workers' interface to aid in drafting and refining their comments. Experts rated the writing quality and labeling validity of CoCoLoFa as high and reliable. BERT-based models fine-tuned using CoCoLoFa achieved the highest fallacy detection (F1=0.86) and classification (F1=0.87) performance on its test set, outperforming the state-of-the-art LLMs. Our work shows that combining crowdsourcing and LLMs enables us to more effectively construct datasets for complex linguistic phenomena that crowd workers find challenging to produce on their own.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03414v2">Logistic Regression makes small LLMs strong and explainable "tens-of-shot" classifiers</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 48 pages, 24 figures
    </div>
    <details class="paper-abstract">
      For simple classification tasks, we show that users can benefit from the advantages of using small, local, generative language models instead of large commercial models without a trade-off in performance or introducing extra labelling costs. These advantages, including those around privacy, availability, cost, and explainability, are important both in commercial applications and in the broader democratisation of AI. Through experiments on 17 sentence classification tasks (2-4 classes), we show that penalised logistic regression on the embeddings from a small LLM equals (and usually betters) the performance of a large LLM in the "tens-of-shot" regime. This requires no more labelled instances than are needed to validate the performance of the large LLM. Finally, we extract stable and sensible explanations for classification decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19874v2">Is In-Context Learning Sufficient for Instruction Following in LLMs?</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Preprint. Code at https://github.com/tml-epfl/icl-alignment
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) allows LLMs to learn from examples without changing their weights: this is a particularly promising capability for long-context LLMs that can potentially learn from many examples. Recently, Lin et al. (2024) proposed URIAL, a method using only three in-context examples to align base LLMs, achieving non-trivial instruction following performance. In this work, we show that, while effective, ICL alignment with URIAL still underperforms compared to instruction fine-tuning on the established benchmark MT-Bench, especially with more capable base LLMs. We then uncover the most relevant elements for successful in-context alignment, finding the crucial role of the decoding parameters. Based on these insights, we show that the approach of URIAL can indeed be improved by adding high-quality, potentially carefully selected via greedy search, demonstrations in context, getting closer to the performance of instruct models. Finally, we provide the first, to our knowledge, systematic comparison of ICL and instruction fine-tuning (IFT) for instruction following in the low data regime, where ICL can be a viable alternative to IFT. Overall, our work advances the understanding of ICL as an alignment technique and its relationship to IFT. We provide our code at https://github.com/tml-epfl/icl-alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.03279v3">Lifelong Knowledge Editing for LLMs with Retrieval-Augmented Continuous Prompt Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 16 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Model editing aims to correct outdated or erroneous knowledge in large language models (LLMs) without the need for costly retraining. Lifelong model editing is the most challenging task that caters to the continuous editing requirements of LLMs. Prior works primarily focus on single or batch editing; nevertheless, these methods fall short in lifelong editing scenarios due to catastrophic knowledge forgetting and the degradation of model performance. Although retrieval-based methods alleviate these issues, they are impeded by slow and cumbersome processes of integrating the retrieved knowledge into the model. In this work, we introduce RECIPE, a RetriEval-augmented ContInuous Prompt lEarning method, to boost editing efficacy and inference efficiency in lifelong learning. RECIPE first converts knowledge statements into short and informative continuous prompts, prefixed to the LLM's input query embedding, to efficiently refine the response grounded on the knowledge. It further integrates the Knowledge Sentinel (KS) that acts as an intermediary to calculate a dynamic threshold, determining whether the retrieval repository contains relevant knowledge. Our retriever and prompt encoder are jointly trained to achieve editing properties, i.e., reliability, generality, and locality. In our experiments, RECIPE is assessed extensively across multiple LLMs and editing datasets, where it achieves superior editing performance. RECIPE also demonstrates its capability to maintain the overall performance of LLMs alongside showcasing fast editing and inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08495v2">Investigating LLMs as Voting Assistants via Contextual Augmentation: A Case Study on the European Parliament Elections 2024</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 accepted to EMNLP 2024 as a short paper
    </div>
    <details class="paper-abstract">
      In light of the recent 2024 European Parliament elections, we are investigating if LLMs can be used as Voting Advice Applications (VAAs). We audit MISTRAL and MIXTRAL models and evaluate their accuracy in predicting the stance of political parties based on the latest "EU and I" voting assistance questionnaire. Furthermore, we explore alternatives to improve models' performance by augmenting the input context via Retrieval-Augmented Generation (RAG) relying on web search, and Self-Reflection using staged conversations that aim to re-collect relevant content from the model's internal memory. We find that MIXTRAL is highly accurate with an 82% accuracy on average with a significant performance disparity across different political groups (50-95%). Augmenting the input context with expert-curated information can lead to a significant boost of approx. 9%, which remains an open challenge for automated RAG approaches, even considering curated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.20279v3">LUQ: Long-text Uncertainty Quantification for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capability in a variety of NLP tasks. However, LLMs are also prone to generate nonfactual content. Uncertainty Quantification (UQ) is pivotal in enhancing our understanding of a model's confidence on its generation, thereby aiding in the mitigation of nonfactual outputs. Existing research on UQ predominantly targets short text generation, typically yielding brief, word-limited responses. However, real-world applications frequently necessitate much longer responses. Our study first highlights the limitations of current UQ methods in handling long text generation. We then introduce \textsc{Luq} and its two variations, a series of novel sampling-based UQ approaches specifically designed for long text. Our findings reveal that \textsc{Luq} outperforms existing baseline methods in correlating with the model's factuality scores (negative coefficient of -0.85 observed for Gemini Pro). To further improve the factuality of LLM responses, we propose \textsc{Luq-Ensemble}, a method that ensembles responses from multiple models and selects the response with the lowest uncertainty. The ensembling method greatly improves the response factuality upon the best standalone LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03234v1">Showing LLM-Generated Code Selectively Based on Confidence of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive abilities in code generation, but they may generate erroneous programs. Reading a program takes ten times longer than writing it. Showing these erroneous programs to developers will waste developers' energies and introduce security risks to software. To address the above limitations, we propose HonestCoder, a novel LLM-based code generation approach. HonestCoder selectively shows the generated programs to developers based on LLMs' confidence. The confidence provides valuable insights into the correctness of generated programs. To achieve this goal, we propose a novel approach to estimate LLMs' confidence in code generation. It estimates confidence by measuring the multi-modal similarity between LLMs-generated programs. We collect and release a multilingual benchmark named TruthCodeBench, which consists of 2,265 samples and covers two popular programming languages (i.e., Python and Java). We apply HonestCoder to four popular LLMs (e.g., DeepSeek-Coder and Code Llama) and evaluate it on TruthCodeBench. Based on the experiments, we obtain the following insights. (1) HonestCoder can effectively estimate LLMs' confidence and accurately determine the correctness of generated programs. For example, HonestCoder outperforms the state-of-the-art baseline by 27.79% in AUROC and 63.74% in AUCPR. (2) HonestCoder can decrease the number of erroneous programs shown to developers. Compared to eight baselines, it can show more correct programs and fewer erroneous programs to developers. (3) Compared to showing code indiscriminately, HonestCoder only adds slight time overhead (approximately 0.4 seconds per requirement). (4) We discuss future directions to facilitate the application of LLMs in software development. We hope this work can motivate broad discussions about measuring the reliability of LLMs' outputs in performing code-related tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.14672v2">Middleware for LLMs: Tools Are Instrumental for Language Agents in Complex Environments</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP'2024; 18 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      The applications of large language models (LLMs) have expanded well beyond the confines of text processing, signaling a new era where LLMs are envisioned as generalist agents capable of operating within complex environments. These environments are often highly expansive, making it impossible for the LLM to process them within its short-term memory. Motivated by recent research on extending the capabilities of LLMs with tools, we seek to investigate the intriguing potential of tools to augment LLMs in handling such complexity by introducing a novel class of tools, termed middleware, to aid in the proactive exploration within these massive environments. Such specialized tools can serve as a middleware layer shielding the LLM from environmental complexity. In two representative complex environments -- knowledge bases (KBs) and databases -- we demonstrate the significant potential of augmenting language agents with tools in complex environments. Notably, equipped with the middleware, GPT-4 achieves 2.8X the performance of the best baseline in tasks requiring access to database content and 2.2X in KB tasks. Our findings illuminate the path for advancing language agents in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03185v1">EXAQ: Exponent Aware Quantization For LLMs Acceleration</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Quantization has established itself as the primary approach for decreasing the computational and storage expenses associated with Large Language Models (LLMs) inference. The majority of current research emphasizes quantizing weights and activations to enable low-bit general-matrix-multiply (GEMM) operations, with the remaining non-linear operations executed at higher precision. In our study, we discovered that following the application of these techniques, the primary bottleneck in LLMs inference lies in the softmax layer. The softmax operation comprises three phases: exponent calculation, accumulation, and normalization, Our work focuses on optimizing the first two phases. We propose an analytical approach to determine the optimal clipping value for the input to the softmax function, enabling sub-4-bit quantization for LLMs inference. This method accelerates the calculations of both $e^x$ and $\sum(e^x)$ with minimal to no accuracy degradation. For example, in LLaMA1-30B, we achieve baseline performance with 2-bit quantization on the well-known "Physical Interaction: Question Answering" (PIQA) dataset evaluation. This ultra-low bit quantization allows, for the first time, an acceleration of approximately 4x in the accumulation phase. The combination of accelerating both $e^x$ and $\sum(e^x)$ results in a 36.9% acceleration in the softmax operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.04224v4">Aligners: Decoupling LLMs and Alignment</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Short version accepted as a Tiny Paper at the International Conference on Learning Representations (ICLR) 2024. Long version accepted to the Conference on Empirical Methods in Natural Language Processing (EMNLP) 2024 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) need to be aligned with human expectations to ensure their safety and utility in most applications. Alignment is challenging, costly, and needs to be repeated for every LLM and alignment criterion. We propose to decouple LLMs and alignment by training aligner models that can be used to align any LLM for a given criteria on an as-needed basis, thus also reducing the potential negative impacts of alignment on performance. Our recipe for training the aligner models solely relies on synthetic data generated with a (prompted) LLM and can be easily adjusted for a variety of alignment criteria. We use the same synthetic data to train inspectors, binary miss-alignment classification models to guide a "squad" of multiple aligners. Our empirical results demonstrate consistent improvements when applying aligner squad to various LLMs, including chat-aligned models, across several instruction-following and red-teaming datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14790v2">Step-by-Step Reasoning to Solve Grid Puzzles: Where do LLMs Falter?</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted at EMNLP 2024 Main
    </div>
    <details class="paper-abstract">
      Solving grid puzzles involves a significant amount of logical reasoning. Hence, it is a good domain to evaluate the reasoning capability of a model which can then guide us to improve the reasoning ability of models. However, most existing works evaluate only the final predicted answer of a puzzle, without delving into an in-depth analysis of the LLMs' reasoning chains (such as where they falter) or providing any finer metrics to evaluate them. Since LLMs may rely on simple heuristics or artifacts to predict the final answer, it is crucial to evaluate the generated reasoning chain beyond overall correctness measures, for accurately evaluating the reasoning abilities of LLMs. To this end, we first develop GridPuzzle, an evaluation dataset comprising 274 grid-based puzzles with different complexities. Second, we propose a new error taxonomy derived from manual analysis of reasoning chains from LLMs including GPT-4, Claude-3, Gemini, Mistral, and Llama-2. Then, we develop an LLM-based framework for large-scale subjective evaluation (i.e., identifying errors) and an objective metric, PuzzleEval, to evaluate the correctness of reasoning chains. Evaluating reasoning chains from LLMs leads to several interesting findings. We further show that existing prompting methods used for enhancing models' reasoning abilities do not improve performance on GridPuzzle. This highlights the importance of understanding fine-grained errors and presents a challenge for future research to enhance LLMs' puzzle-solving abilities by developing methods that address these errors. Data and source code are available at https://github.com/Mihir3009/GridPuzzle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03138v1">Can LLMs Generate Diverse Molecules? Towards Alignment with Structural Diversity</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated impressive performance in generating molecular structures as drug candidates, which offers significant potential to accelerate drug discovery. However, the current LLMs overlook a critical requirement for drug discovery: proposing a diverse set of molecules. This diversity is essential for improving the chances of finding a viable drug, as it provides alternative molecules that may succeed where others fail in wet-lab or clinical validations. Despite such a need for diversity, the LLMs often output structurally similar molecules from a given prompt. While decoding schemes like beam search may enhance textual diversity, this often does not align with molecular structural diversity. In response, we propose a new method for fine-tuning molecular generative LLMs to autoregressively generate a set of structurally diverse molecules, where each molecule is generated by conditioning on the previously generated molecules. Our approach consists of two stages: (1) supervised fine-tuning to adapt LLMs to autoregressively generate molecules in a sequence and (2) reinforcement learning to maximize structural diversity within the generated molecules. Our experiments show that (1) our fine-tuning approach enables the LLMs to better discover diverse molecules compared to existing decoding schemes and (2) our fine-tuned model outperforms other representative LLMs in generating diverse molecules, including the ones fine-tuned on chemical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02736v2">Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has been widely utilized as an evaluation method in various benchmarks and served as supervised rewards in model training. However, despite their excellence in many domains, potential issues are under-explored, undermining their reliability and the scope of their utility. Therefore, we identify 12 key potential biases and propose a new automated bias quantification framework-CALM-which systematically quantifies and analyzes each type of bias in LLM-as-a-Judge by using automated and principle-guided modification. Our experiments cover multiple popular language models, and the results indicate that while advanced models have achieved commendable overall performance, significant biases persist in certain specific tasks. Empirical results suggest that there remains room for improvement in the reliability of LLM-as-a-Judge. Moreover, we also discuss the explicit and implicit influence of these biases and give some suggestions for the reliable application of LLM-as-a-Judge. Our work highlights the need for stakeholders to address these issues and remind users to exercise caution in LLM-as-a-Judge applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00216v2">Is Factuality Enhancement a Free Lunch For LLMs? Better Factuality Can Lead to Worse Context-Faithfulness</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      As the modern tools of choice for text understanding and generation, large language models (LLMs) are expected to accurately output answers by leveraging the input context. This requires LLMs to possess both context-faithfulness and factual accuracy. Extensive efforts have been made to enable better outputs from LLMs by mitigating hallucinations through factuality enhancement methods. However, they also pose risks of hindering context-faithfulness, as factuality enhancement can lead LLMs to become overly confident in their parametric knowledge, causing them to overlook the relevant input context. In this work, we argue that current factuality enhancement methods can significantly undermine the context-faithfulness of LLMs. We first revisit the current factuality enhancement methods and evaluate their effectiveness in enhancing factual accuracy. Next, we evaluate their performance on knowledge editing tasks to assess the potential impact on context-faithfulness. The experimental results reveal that while these methods may yield inconsistent improvements in factual accuracy, they also cause a more severe decline in context-faithfulness, with the largest decrease reaching a striking 69.7\%. To explain these declines, we analyze the hidden states and logit distributions for the tokens representing new knowledge and parametric knowledge respectively, highlighting the limitations of current approaches. Our finding highlights the complex trade-offs inherent in enhancing LLMs. Therefore, we recommend that more research on LLMs' factuality enhancement make efforts to reduce the sacrifice of context-faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03111v1">LoRC: Low-Rank Compression for LLMs KV Cache with a Progressive Compression Strategy</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The Key-Value (KV) cache is a crucial component in serving transformer-based autoregressive large language models (LLMs), enabling faster inference by storing previously computed KV vectors. However, its memory consumption scales linearly with sequence length and batch size, posing a significant bottleneck in LLM deployment. Existing approaches to mitigate this issue include: (1) efficient attention variants integrated in upcycling stages, which requires extensive parameter tuning thus unsuitable for pre-trained LLMs; (2) KV cache compression at test time, primarily through token eviction policies, which often overlook inter-layer dependencies and can be task-specific. This paper introduces an orthogonal approach to KV cache compression. We propose a low-rank approximation of KV weight matrices, allowing for plug-in integration with existing transformer-based LLMs without model retraining. To effectively compress KV cache at the weight level, we adjust for layerwise sensitivity and introduce a progressive compression strategy, which is supported by our theoretical analysis on how compression errors accumulate in deep networks. Our method is designed to function without model tuning in upcycling stages or task-specific profiling in test stages. Extensive experiments with LLaMA models ranging from 8B to 70B parameters across various tasks show that our approach significantly reduces the GPU memory footprint while maintaining performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03203v2">TheoremLlama: Transforming General-Purpose LLMs into Lean4 Experts</a></div>
    <div class="paper-meta">
      📅 2024-10-04
    </div>
    <details class="paper-abstract">
      Proving mathematical theorems using computer-verifiable formal languages like Lean significantly impacts mathematical reasoning. One approach to formal theorem proving involves generating complete proofs using Large Language Models (LLMs) based on Natural Language (NL) proofs. However, due to the scarcity of aligned NL and Formal Language (FL) theorem-proving data most modern LLMs exhibit suboptimal performance.This scarcity results in a paucity of methodologies for training LLMs and techniques to fully utilize their capabilities in composing formal proofs. To address these challenges, this paper proposes TheoremLlama, an end-to-end framework that trains a general-purpose LLM to be a Lean4 expert. TheoremLlama includes NL-FL dataset generation and bootstrapping method to obtain aligned dataset, curriculum learning and block training techniques to train the model, and iterative proof writing method to write Lean4 proofs that work together synergistically. Using the dataset generation method in TheoremLlama, we provide Open Bootstrapped Theorems (OBT), an NL-FL aligned and bootstrapped dataset. Our novel NL-FL bootstrapping method, where NL proofs are integrated into Lean4 code for training datasets, leverages the NL reasoning ability of LLMs for formal reasoning. The TheoremLlama framework achieves cumulative accuracies of 36.48% and 33.61% on MiniF2F-Valid and Test datasets respectively, surpassing the GPT-4 baseline of 22.95% and 25.41%. Our code, model checkpoints, and the generated dataset is published in GitHub
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.05020v4">Is this the real life? Is this just fantasy? The Misleading Success of Simulating Social Interactions With LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLM) have enabled richer social simulations, allowing for the study of various social phenomena. However, most recent work has used a more omniscient perspective on these simulations (e.g., single LLM to generate all interlocutors), which is fundamentally at odds with the non-omniscient, information asymmetric interactions that involve humans and AI agents in the real world. To examine these differences, we develop an evaluation framework to simulate social interactions with LLMs in various settings (omniscient, non-omniscient). Our experiments show that LLMs perform better in unrealistic, omniscient simulation settings but struggle in ones that more accurately reflect real-world conditions with information asymmetry. Our findings indicate that addressing information asymmetry remains a fundamental challenge for LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03061v1">DocKD: Knowledge Distillation from LLMs for Open-World Document Understanding Models</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted to EMNLP 2024
    </div>
    <details class="paper-abstract">
      Visual document understanding (VDU) is a challenging task that involves understanding documents across various modalities (text and image) and layouts (forms, tables, etc.). This study aims to enhance generalizability of small VDU models by distilling knowledge from LLMs. We identify that directly prompting LLMs often fails to generate informative and useful data. In response, we present a new framework (called DocKD) that enriches the data generation process by integrating external document knowledge. Specifically, we provide an LLM with various document elements like key-value pairs, layouts, and descriptions, to elicit open-ended answers. Our experiments show that DocKD produces high-quality document annotations and surpasses the direct knowledge distillation approach that does not leverage external document knowledge. Moreover, student VDU models trained with solely DocKD-generated data are not only comparable to those trained with human-annotated data on in-domain tasks but also significantly excel them on out-of-domain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.02425v1">LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services</a></div>
    <div class="paper-meta">
      📅 2024-10-04
      | 💬 Accepted to the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '24)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are rapidly growing in popularity, LLM inference services must be able to serve requests from thousands of users while satisfying performance requirements. The performance of an LLM inference service is largely determined by the hardware onto which it is deployed, but understanding of which hardware will deliver on performance requirements remains challenging. In this work we present LLM-Pilot - a first-of-its-kind system for characterizing and predicting performance of LLM inference services. LLM-Pilot performs benchmarking of LLM inference services, under a realistic workload, across a variety of GPUs, and optimizes the service configuration for each considered GPU to maximize performance. Finally, using this characterization data, LLM-Pilot learns a predictive model, which can be used to recommend the most cost-effective hardware for a previously unseen LLM. Compared to existing methods, LLM-Pilot can deliver on performance requirements 33% more frequently, whilst reducing costs by 60% on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14690v1">Rethinking VLMs and LLMs for Image Classification</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Visual Language Models (VLMs) are now increasingly being merged with Large Language Models (LLMs) to enable new capabilities, particularly in terms of improved interactivity and open-ended responsiveness. While these are remarkable capabilities, the contribution of LLMs to enhancing the longstanding key problem of classifying an image among a set of choices remains unclear. Through extensive experiments involving seven models, ten visual understanding datasets, and multiple prompt variations per dataset, we find that, for object and scene recognition, VLMs that do not leverage LLMs can achieve better performance than VLMs that do. Yet at the same time, leveraging LLMs can improve performance on tasks requiring reasoning and outside knowledge. In response to these challenges, we propose a pragmatic solution: a lightweight fix involving a relatively small LLM that efficiently routes visual tasks to the most suitable model for the task. The LLM router undergoes training using a dataset constructed from more than 2.5 million examples of pairs of visual task and model accuracy. Our results reveal that this lightweight fix surpasses or matches the accuracy of state-of-the-art alternatives, including GPT-4V and HuggingGPT, while improving cost-effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00242v3">DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Update DeFT-v3 with more ablation studies. DeFT-v1 was accepted by ICLR'24 AGI Workshop ( https://openreview.net/forum?id=HqfLHoX8bR ). Code will be released soon
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly employed for complex tasks that process multiple generation calls in a tree structure with shared prefixes of tokens, including few-shot prompting, multi-step reasoning, speculative decoding, etc. However, existing inference systems for tree-based applications are inefficient due to improper partitioning of queries and KV cache during attention calculation. This leads to two main issues: (1) a lack of memory access (IO) reuse for KV cache of shared prefixes, and (2) poor load balancing.As a result, there is redundant KV cache IO between GPU global memory and shared memory, along with low GPU utilization. To address these challenges, we propose DeFT(Decoding with Flash Tree-Attention), a hardware-efficient attention algorithm with prefix-aware and load-balanced KV cache partitions. DeFT reduces the number of read/write operations of KV cache during attention calculation through KV-Guided Grouping, a method that avoids repeatedly loading KV cache of shared prefixes in attention computation. Additionally, we propose Flattened Tree KV Splitting, a mechanism that ensures even distribution of the KV cache across partitions with little computation redundancy, enhancing GPU utilization during attention computations. By reducing 73-99 KV cache IO and nearly 100 IO for partial results during attention calculation, DeFT achieves up to 2.52/3.82x speedup in the end-to-end/attention latency across three practical tree-based workloads compared to state-of-the-art attention algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01586v4">TrustAgent: Towards Safe and Trustworthy LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 In EMNLP 2024
    </div>
    <details class="paper-abstract">
      The rise of LLM-based agents shows great potential to revolutionize task planning, capturing significant attention. Given that these agents will be integrated into high-stake domains, ensuring their reliability and safety is crucial. This paper presents an Agent-Constitution-based agent framework, TrustAgent, with a particular focus on improving the LLM-based agent safety. The proposed framework ensures strict adherence to the Agent Constitution through three strategic components: pre-planning strategy which injects safety knowledge to the model before plan generation, in-planning strategy which enhances safety during plan generation, and post-planning strategy which ensures safety by post-planning inspection. Our experimental results demonstrate that the proposed framework can effectively enhance an LLM agent's safety across multiple domains by identifying and mitigating potential dangers during the planning. Further analysis reveals that the framework not only improves safety but also enhances the helpfulness of the agent. Additionally, we highlight the importance of the LLM reasoning ability in adhering to the Constitution. This paper sheds light on how to ensure the safe integration of LLM-based agents into human-centric environments. Data and code are available at https://github.com/agiresearch/TrustAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09435v2">MUSCLE: A Model Update Strategy for Compatible LLM Evolution</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are regularly updated to enhance performance, typically through changes in data or architecture. Within the update process, developers often prioritize improving overall performance metrics, paying less attention to maintaining compatibility with earlier model versions. Instance-level degradation (instance regression) of performance from one model version to the next can interfere with a user's mental model of the capabilities of a particular language model. Users having to adapt their mental model with every update can lead to dissatisfaction, especially when the new model has degraded compared to a prior version for a known use case (model update regression). We find that when pretrained LLM base models are updated, fine-tuned user-facing downstream task adapters experience negative flips -- previously correct instances are now predicted incorrectly. We observe model update regression between different model versions on a diverse set of tasks and models, even when the downstream task training procedures remain identical. We argue for the importance of maintaining model update compatibility during updates, and present evaluation metrics designed specifically for generative tasks, while also being applicable to discriminative tasks. We propose a training strategy to minimize the extent of instance regression in model updates, involving training of a compatibility adapter that can enhance task fine-tuned language models. We show negative flips reduce by up to 40% e.g. when updating Llama 1 to Llama 2 with our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.00811v3">Cognitive Bias in Decision-Making with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential as tools to support an expanding range of decision-making tasks. Given their training on human (created) data, LLMs have been shown to inherit societal biases against protected groups, as well as be subject to bias functionally resembling cognitive bias. Human-like bias can impede fair and explainable decisions made with LLM assistance. Our work introduces BiasBuster, a framework designed to uncover, evaluate, and mitigate cognitive bias in LLMs, particularly in high-stakes decision-making tasks. Inspired by prior research in psychology and cognitive science, we develop a dataset containing 13,465 prompts to evaluate LLM decisions on different cognitive biases (e.g., prompt-induced, sequential, inherent). We test various bias mitigation strategies, while proposing a novel method utilizing LLMs to debias their own human-like cognitive bias within prompts. Our analysis provides a comprehensive picture of the presence and effects of cognitive bias across commercial and open-source models. We demonstrate that our selfhelp debiasing effectively mitigates model answers that display patterns akin to human cognitive bias without having to manually craft examples for each bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19502v2">Hierarchical Deconstruction of LLM Reasoning: A Graph-Based Framework for Analyzing Knowledge Utilization</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 published at EMNLP 2024; code is available at https://github.com/kaistAI/knowledge-reasoning
    </div>
    <details class="paper-abstract">
      Despite the advances in large language models (LLMs), how they use their knowledge for reasoning is not yet well understood. In this study, we propose a method that deconstructs complex real-world questions into a graph, representing each question as a node with predecessors of background knowledge needed to solve the question. We develop the DepthQA dataset, deconstructing questions into three depths: (i) recalling conceptual knowledge, (ii) applying procedural knowledge, and (iii) analyzing strategic knowledge. Based on a hierarchical graph, we quantify forward discrepancy, a discrepancy in LLM performance on simpler sub-problems versus complex questions. We also measure backward discrepancy where LLMs answer complex questions but struggle with simpler ones. Our analysis shows that smaller models exhibit more discrepancies than larger models. Distinct patterns of discrepancies are observed across model capacity and possibility of training data memorization. Additionally, guiding models from simpler to complex questions through multi-turn interactions improves performance across model sizes, highlighting the importance of structured intermediate steps in knowledge reasoning. This work enhances our understanding of LLM reasoning and suggests ways to improve their problem-solving abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16090v2">EPO: Hierarchical LLM Agents with Environment Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Long-horizon decision-making tasks present significant challenges for LLM-based agents due to the need for extensive planning over multiple steps. In this paper, we propose a hierarchical framework that decomposes complex tasks into manageable subgoals, utilizing separate LLMs for subgoal prediction and low-level action generation. To address the challenge of creating training signals for unannotated datasets, we develop a reward model that leverages multimodal environment feedback to automatically generate reward signals. We introduce Environment Preference Optimization (EPO), a novel method that generates preference signals from the environment's feedback and uses them to train LLM-based agents. Extensive experiments on ALFRED demonstrate the state-of-the-art performance of our framework, achieving first place on the ALFRED public leaderboard and showcasing its potential to improve long-horizon decision-making in diverse environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02959v1">Coal Mining Question Answering with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel approach to coal mining question answering (QA) using large language models (LLMs) combined with tailored prompt engineering techniques. Coal mining is a complex, high-risk industry where accurate, context-aware information is critical for safe and efficient operations. Current QA systems struggle to handle the technical and dynamic nature of mining-related queries. To address these challenges, we propose a multi-turn prompt engineering framework designed to guide LLMs, such as GPT-4, in answering coal mining questions with higher precision and relevance. By breaking down complex queries into structured components, our approach allows LLMs to process nuanced technical information more effectively. We manually curated a dataset of 500 questions from real-world mining scenarios and evaluated the system's performance using both accuracy (ACC) and GPT-4-based scoring metrics. Experiments comparing ChatGPT, Claude2, and GPT-4 across baseline, chain-of-thought (CoT), and multi-turn prompting methods demonstrate that our method significantly improves both accuracy and contextual relevance, with an average accuracy improvement of 15-18\% and a notable increase in GPT-4 scores. The results show that our prompt-engineering approach provides a robust, adaptable solution for domain-specific question answering in high-stakes environments like coal mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02958v1">AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 47 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Automated machine learning (AutoML) accelerates AI development by automating tasks in the development pipeline, such as optimal model search and hyperparameter tuning. Existing AutoML systems often require technical expertise to set up complex tools, which is in general time-consuming and requires a large amount of human effort. Therefore, recent works have started exploiting large language models (LLM) to lessen such burden and increase the usability of AutoML frameworks via a natural language interface, allowing non-expert users to build their data-driven solutions. These methods, however, are usually designed only for a particular process in the AI development pipeline and do not efficiently use the inherent capacity of the LLMs. This paper proposes AutoML-Agent, a novel multi-agent framework tailored for full-pipeline AutoML, i.e., from data retrieval to model deployment. AutoML-Agent takes user's task descriptions, facilitates collaboration between specialized LLM agents, and delivers deployment-ready models. Unlike existing work, instead of devising a single plan, we introduce a retrieval-augmented planning strategy to enhance exploration to search for more optimal plans. We also decompose each plan into sub-tasks (e.g., data preprocessing and neural network design) each of which is solved by a specialized agent we build via prompting executing in parallel, making the search process more efficient. Moreover, we propose a multi-stage verification to verify executed results and guide the code generation LLM in implementing successful solutions. Extensive experiments on seven downstream tasks using fourteen datasets show that AutoML-Agent achieves a higher success rate in automating the full AutoML process, yielding systems with good performance throughout the diverse domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02950v1">LLMCO2: Advancing Accurate Carbon Footprint Prediction for LLM Inferences</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 9 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Throughout its lifecycle, a large language model (LLM) generates a substantially larger carbon footprint during inference than training. LLM inference requests vary in batch size, prompt length, and token generation number, while cloud providers employ different GPU types and quantities to meet diverse service-level objectives for accuracy and latency. It is crucial for both users and cloud providers to have a tool that quickly and accurately estimates the carbon impact of LLM inferences based on a combination of inference request and hardware configurations before execution. Estimating the carbon footprint of LLM inferences is more complex than training due to lower and highly variable model FLOPS utilization, rendering previous equation-based models inaccurate. Additionally, existing machine learning (ML) prediction methods either lack accuracy or demand extensive training data, as they inadequately handle the distinct prefill and decode phases, overlook hardware-specific features, and inefficiently sample uncommon inference configurations. We introduce \coo, a graph neural network (GNN)-based model that greatly improves the accuracy of LLM inference carbon footprint predictions compared to previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02879v1">Position: LLM Unlearning Benchmarks are Weak Measures of Progress</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Unlearning methods have the potential to improve the privacy and safety of large language models (LLMs) by removing sensitive or harmful information post hoc. The LLM unlearning research community has increasingly turned toward empirical benchmarks to assess the effectiveness of such methods. In this paper, we find that existing benchmarks provide an overly optimistic and potentially misleading view on the effectiveness of candidate unlearning methods. By introducing simple, benign modifications to a number of popular benchmarks, we expose instances where supposedly unlearned information remains accessible, or where the unlearning process has degraded the model's performance on retained information to a much greater extent than indicated by the original benchmark. We identify that existing benchmarks are particularly vulnerable to modifications that introduce even loose dependencies between the forget and retain information. Further, we show that ambiguity in unlearning targets in existing benchmarks can easily lead to the design of methods that overfit to the given test queries. Based on our findings, we urge the community to be cautious when interpreting benchmark results as reliable measures of progress, and we provide several recommendations to guide future LLM unlearning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00023v2">Preble: Efficient Distributed Prompt Scheduling for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Prompts to large language models (LLMs) have evolved beyond simple user questions. For LLMs to solve complex problems, today's practices are to include domain-specific instructions, illustration of tool usages, and/or long context such as textbook chapters in prompts. As such, many parts of prompts are repetitive across requests. Recent works propose to cache and reuse KV state of prompts. However, they are all confined to a single-GPU optimization, while production LLM serving systems are distributed by nature. This paper proposes Preble, the first distributed LLM serving platform that targets and optimizes for prompt sharing. We designed a distributed scheduling system that co-optimizes KV state reuse and computation load-balancing with a new scheduling algorithm and a hierarchical scheduling mechanism. Our evaluation of Preble with real workloads and request arrival patterns on two open-source LLMs shows that Preble outperforms the SOTA serving systems by 1.5X to 14.5X on average latency and 2X to 10X on p99 latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02725v1">Adaptive Inference-Time Compute: LLMs Can Predict if They Can Do Better, Even Mid-Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Inference-time computation is a powerful paradigm to enhance the performance of large language models (LLMs), with Best-of-N sampling being a widely used technique. However, this method is computationally expensive, requiring both (1) an external reward model and (2) the generation of multiple samples. In this work, we introduce a new generative self-evaluation scheme designed to adaptively reduce the number of generated samples while maintaining or even improving performance. We use a generative reward model formulation, allowing the LLM to predict mid-generation the probability that restarting the generation will yield a better response. These predictions are obtained without an external reward model and can be used to decide whether or not to generate more samples, prune unpromising samples early on, or to pick the best sample. This capability is very inexpensive as it involves generating a single predefined token. Trained using a dataset constructed with real unfiltered LMSYS user prompts, Llama 3.1 8B's win rate against GPT-4 on AlpacaEval increases from 21% to 34% with 16 samples and math performance on GSM8K improves from 84% to 91%. By sampling only when the LLM determines that it is beneficial to do so and adaptively adjusting temperature annealing, we demonstrate that 74% of the improvement from using 16 samples can be achieved with only 1.2 samples on average. We further demonstrate that 50-75% of samples can be pruned early in generation with minimal degradation in performance. Overall, our methods enable more efficient and scalable compute utilization during inference for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12836v1">EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Given the steep learning curve of professional 3D software and the time-consuming process of managing large 3D assets, language-guided 3D scene editing has significant potential in fields such as virtual reality, augmented reality, and gaming. However, recent approaches to language-guided 3D scene editing either require manual interventions or focus only on appearance modifications without supporting comprehensive scene layout changes. In response, we propose Edit-Room, a unified framework capable of executing a variety of layout edits through natural language commands, without requiring manual intervention. Specifically, EditRoom leverages Large Language Models (LLMs) for command planning and generates target scenes using a diffusion-based method, enabling six types of edits: rotate, translate, scale, replace, add, and remove. To address the lack of data for language-guided 3D scene editing, we have developed an automatic pipeline to augment existing 3D scene synthesis datasets and introduced EditRoom-DB, a large-scale dataset with 83k editing pairs, for training and evaluation. Our experiments demonstrate that our approach consistently outperforms other baselines across all metrics, indicating higher accuracy and coherence in language-guided scene layout editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.12683v2">Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted at Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      The vast majority of today's large language models (LLMs) are English-centric, having been pretrained predominantly on English text. Yet, in order to meet user expectations, models need to be able to respond appropriately in multiple languages once deployed in downstream applications. This requires strong cross-lingual transfer abilities. In this work, we investigate the minimal amount of multilinguality required during finetuning to elicit cross-lingual generalisation in English-centric LLMs. In experiments across four LLMs, we find that multilingual instruction tuning with as few as two to three languages is both necessary and sufficient to elicit effective cross-lingual generalisation, with the limiting factor being the degree to which a target language is seen during pretraining. Evaluations on five different tasks further reveal that multilingual instruction tuning is most beneficial for generative tasks that assume input/output language agreement, such as in chat settings, while being of less importance for highly structured classification-style tasks. Our code and data is available at https://github.com/ZurichNLP/multilingual-instruction-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18725v2">Jailbreaking LLMs with Arabic Transliteration and Arabizi</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted by EMNLP 2024
    </div>
    <details class="paper-abstract">
      This study identifies the potential vulnerabilities of Large Language Models (LLMs) to 'jailbreak' attacks, specifically focusing on the Arabic language and its various forms. While most research has concentrated on English-based prompt manipulation, our investigation broadens the scope to investigate the Arabic language. We initially tested the AdvBench benchmark in Standardized Arabic, finding that even with prompt manipulation techniques like prefix injection, it was insufficient to provoke LLMs into generating unsafe content. However, when using Arabic transliteration and chatspeak (or arabizi), we found that unsafe content could be produced on platforms like OpenAI GPT-4 and Anthropic Claude 3 Sonnet. Our findings suggest that using Arabic and its various forms could expose information that might remain hidden, potentially increasing the risk of jailbreak attacks. We hypothesize that this exposure could be due to the model's learned connection to specific words, highlighting the need for more comprehensive safety training across all language forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02683v1">DailyDilemmas: Revealing Value Preferences of LLMs with Quandaries of Daily Life</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      As we increasingly seek guidance from LLMs for decision-making in daily life, many of these decisions are not clear-cut and depend significantly on the personal values and ethical standards of the users. We present DailyDilemmas, a dataset of 1,360 moral dilemmas encountered in everyday life. Each dilemma includes two possible actions and with each action, the affected parties and human values invoked. Based on these dilemmas, we consolidated a set of human values across everyday topics e.g., interpersonal relationships, workplace, and environmental issues. We evaluated LLMs on these dilemmas to determine what action they will take and the values represented by these actions. Then, we analyzed these values through the lens of five popular theories inspired by sociology, psychology and philosophy. These theories are: World Value Survey, Moral Foundation Theory, Maslow's Hierarchy of Needs, Aristotle's Virtues, and Plutchik Wheel of Emotion. We find that LLMs are most aligned with the self-expression over survival values in terms of World Value Survey, care over loyalty in Moral Foundation Theory. Interestingly, we find large preferences differences in models for some core values such as truthfulness e.g., Mixtral-8x7B model tends to neglect it by 9.7% while GPT-4-turbo model tends to select it by 9.4%. We also study the recent guidance released by OpenAI (ModelSpec), and Anthropic (Constitutional AI) to understand how their released principles reflect their actual value prioritization when facing nuanced moral reasoning in daily-life settings. We find that end users cannot effectively steer such prioritization using system prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02677v1">CulturalBench: a Robust, Diverse and Challenging Benchmark on Measuring the (Lack of) Cultural Knowledge of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      To make large language models (LLMs) more helpful across diverse cultures, it is essential to have effective cultural knowledge benchmarks to measure and track our progress. Effective benchmarks need to be robust, diverse, and challenging. We introduce CulturalBench: a set of 1,227 human-written and human-verified questions for effectively assessing LLMs' cultural knowledge, covering 45 global regions including the underrepresented ones like Bangladesh, Zimbabwe, and Peru. Questions - each verified by five independent annotators - span 17 diverse topics ranging from food preferences to greeting etiquettes. We evaluate models on two setups: CulturalBench-Easy and CulturalBench-Hard which share the same questions but asked differently. We find that LLMs are sensitive to such difference in setups (e.g., GPT-4o with 27.3% difference). Compared to human performance (92.6% accuracy), CulturalBench-Hard is more challenging for frontier LLMs with the best performing model (GPT-4o) at only 61.5% and the worst (Llama3-8b) at 21.4%. Moreover, we find that LLMs often struggle with tricky questions that have multiple correct answers (e.g., What utensils do the Chinese usually use?), revealing a tendency to converge to a single answer. Our results also indicate that OpenAI GPT-4o substantially outperform other proprietary and open source models in questions related to all but one region (Oceania). Nonetheless, all models consistently underperform on questions related to South America and the Middle East.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11969v3">Does Refusal Training in LLMs Generalize to the Past Tense?</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Update in v3: o1-mini and o1-preview results (on top of GPT-4o and Claude 3.5 Sonnet added in v2). We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense
    </div>
    <details class="paper-abstract">
      Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02657v1">Hate Personified: Investigating the role of LLMs in content moderation</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 17 pages, 6 Figures, 13 Tables, EMNLP'24 Mains
    </div>
    <details class="paper-abstract">
      For subjective tasks such as hate detection, where people perceive hate differently, the Large Language Model's (LLM) ability to represent diverse groups is unclear. By including additional context in prompts, we comprehensively analyze LLM's sensitivity to geographical priming, persona attributes, and numerical information to assess how well the needs of various groups are reflected. Our findings on two LLMs, five languages, and six datasets reveal that mimicking persona-based attributes leads to annotation variability. Meanwhile, incorporating geographical signals leads to better regional alignment. We also find that the LLMs are sensitive to numerical anchors, indicating the ability to leverage community-based flagging efforts and exposure to adversaries. Our work provides preliminary guidelines and highlights the nuances of applying LLMs in culturally sensitive cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v1">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 23 different types of attack/defense methods, and 8 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, a mixed attack, and 10 corresponding defenses across 13 LLM backbones with nearly 90,000 testing cases in total. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13681v2">PARAMANU-AYN: Pretrain from scratch or Continual Pretraining of LLMs for Legal Domain Adaptation?</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      In this paper, we present Paramanu-Ayn, a collection of legal language models trained exclusively on Indian legal case documents. This 97-million-parameter Auto-Regressive (AR) decoder-only model was pretrained from scratch with a context size of 8192 on a single GPU for just 185 hours, achieving an efficient MFU of 41.35. We also developed a legal domain specialized BPE tokenizer. We evaluated our model using perplexity and zero-shot tasks: case judgment prediction with explanation and abstractive case summarization. Paramanu-Ayn outperformed Llama-2 7B and Gemini-Pro in case judgment prediction with explanation task on test accuracy by nearly 2 percentage points, despite being 72 times smaller. In zero-shot abstractive summarization, it surpassed decoder-only LLMs generating fixed-length summaries (5000 tokens) by over 10 percentage points in BLEU and METEOR metrics, and by nearly 4 percentage points in BERTScore. Further evaluations on zero-shot commonsense and mathematical benchmarks showed that Paramanu-Ayn excelled despite being trained exclusively on legal documents, outperforming Llama-1, Llama-2, and Falcon on AGIEVAL-AQuA-RAT and AGIEVAL-SAT-Math tasks. We also instruction-tuned our model on 10,763 diverse legal tasks, including legal clause generation, legal drafting, case summarization, etc. The Paramanu-Ayn-instruct model scored above 8 out of 10 in clarity, relevance, completeness, and legal reasoning metrics by GPT-3.5-Turbo. We found that our models, were able to learn drafting knowledge and generalize to draft legal contracts and legal clauses with limited instruction-tuning. Hence, we conclude that for a strong domain-specialized generative language model (such as legal), domain specialized pretraining from scratch is more cost effective, environmentally friendly, and remains competitive with larger models or even better than adapting LLMs for legal domain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02584v1">Towards Implicit Bias Detection and Mitigation in Multi-Agent LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted to EMNLP Findings 2024
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to evolve, they are increasingly being employed in numerous studies to simulate societies and execute diverse social tasks. However, LLMs are susceptible to societal biases due to their exposure to human-generated data. Given that LLMs are being used to gain insights into various societal aspects, it is essential to mitigate these biases. To that end, our study investigates the presence of implicit gender biases in multi-agent LLM interactions and proposes two strategies to mitigate these biases. We begin by creating a dataset of scenarios where implicit gender biases might arise, and subsequently develop a metric to assess the presence of biases. Our empirical analysis reveals that LLMs generate outputs characterized by strong implicit bias associations (>= 50\% of the time). Furthermore, these biases tend to escalate following multi-agent interactions. To mitigate them, we propose two strategies: self-reflection with in-context examples (ICE); and supervised fine-tuning. Our research demonstrates that both methods effectively mitigate implicit biases, with the ensemble of fine-tuning and self-reflection proving to be the most successful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18045v3">Multi-FAct: Assessing Factuality of Multilingual LLMs using FActScore</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Evaluating the factuality of long-form large language model (LLM)-generated text is an important challenge. Recently there has been a surge of interest in factuality evaluation for English, but little is known about the factuality evaluation of multilingual LLMs, specially when it comes to long-form generation. %This paper systematically evaluates multilingual LLMs' factual accuracy across languages and geographic regions. We introduce a simple pipeline for multilingual factuality evaluation, by applying FActScore (Min et al., 2023) for diverse languages. In addition to evaluating multilingual factual generation, we evaluate the factual accuracy of long-form text generation in topics that reflect regional diversity. We also examine the feasibility of running the FActScore pipeline using non-English Wikipedia and provide comprehensive guidelines on multilingual factual evaluation for regionally diverse topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20242v3">BadRobot: Manipulating Embodied LLMs in the Physical World</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 38 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Embodied AI represents systems where AI is integrated into physical entities, enabling them to perceive and interact with their surroundings. Large Language Model (LLM), which exhibits powerful language understanding abilities, has been extensively employed in embodied AI by facilitating sophisticated task planning. However, a critical safety issue remains overlooked: could these embodied LLMs perpetrate harmful behaviors? In response, we introduce BadRobot, a novel attack paradigm aiming to make embodied LLMs violate safety and ethical constraints through typical voice-based user-system interactions. Specifically, three vulnerabilities are exploited to achieve this type of attack: (i) manipulation of LLMs within robotic systems, (ii) misalignment between linguistic outputs and physical actions, and (iii) unintentional hazardous behaviors caused by world knowledge's flaws. Furthermore, we construct a benchmark of various malicious physical action queries to evaluate BadRobot's attack performance. Based on this benchmark, extensive experiments against existing prominent embodied LLM frameworks (e.g., Voxposer, Code as Policies, and ProgPrompt) demonstrate the effectiveness of our BadRobot. Warning: This paper contains harmful AI-generated language and aggressive actions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02511v1">Choices are More Important than Efforts: LLM Enables Efficient Multi-Agent Exploration</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      With expansive state-action spaces, efficient multi-agent exploration remains a longstanding challenge in reinforcement learning. Although pursuing novelty, diversity, or uncertainty attracts increasing attention, redundant efforts brought by exploration without proper guidance choices poses a practical issue for the community. This paper introduces a systematic approach, termed LEMAE, choosing to channel informative task-relevant guidance from a knowledgeable Large Language Model (LLM) for Efficient Multi-Agent Exploration. Specifically, we ground linguistic knowledge from LLM into symbolic key states, that are critical for task fulfillment, in a discriminative manner at low LLM inference costs. To unleash the power of key states, we design Subspace-based Hindsight Intrinsic Reward (SHIR) to guide agents toward key states by increasing reward density. Additionally, we build the Key State Memory Tree (KSMT) to track transitions between key states in a specific task for organized exploration. Benefiting from diminishing redundant explorations, LEMAE outperforms existing SOTA approaches on the challenging benchmarks (e.g., SMAC and MPE) by a large margin, achieving a 10x acceleration in certain scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02506v1">Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Recent advancements in large language model (LLM)-powered agents have shown that collective intelligence can significantly outperform individual capabilities, largely attributed to the meticulously designed inter-agent communication topologies. Though impressive in performance, existing multi-agent pipelines inherently introduce substantial token overhead, as well as increased economic costs, which pose challenges for their large-scale deployments. In response to this challenge, we propose an economical, simple, and robust multi-agent communication framework, termed $\texttt{AgentPrune}$, which can seamlessly integrate into mainstream multi-agent systems and prunes redundant or even malicious communication messages. Technically, $\texttt{AgentPrune}$ is the first to identify and formally define the \textit{communication redundancy} issue present in current LLM-based multi-agent pipelines, and efficiently performs one-shot pruning on the spatial-temporal message-passing graph, yielding a token-economic and high-performing communication topology. Extensive experiments across six benchmarks demonstrate that $\texttt{AgentPrune}$ \textbf{(I)} achieves comparable results as state-of-the-art topologies at merely $\$5.6$ cost compared to their $\$43.7$, \textbf{(II)} integrates seamlessly into existing multi-agent frameworks with $28.1\%\sim72.8\%\downarrow$ token reduction, and \textbf{(III)} successfully defend against two types of agent-based adversarial attacks with $3.5\%\sim10.8\%\uparrow$ performance boost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02486v1">Encryption-Friendly LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 27 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer personalized responses based on user interactions, but this use case raises serious privacy concerns. Homomorphic encryption (HE) is a cryptographic protocol supporting arithmetic computations in encrypted states and provides a potential solution for privacy-preserving machine learning (PPML). However, the computational intensity of transformers poses challenges for applying HE to LLMs. In this work, we propose a modified HE-friendly transformer architecture with an emphasis on inference following personalized (private) fine-tuning. Utilizing LoRA fine-tuning and Gaussian kernels, we achieve significant computational speedups -- 6.94x for fine-tuning and 2.3x for inference -- while maintaining performance comparable to plaintext models. Our findings provide a viable proof of concept for offering privacy-preserving LLM services in areas where data protection is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01242v2">RGD: Multi-LLM Based Agent Debugger via Refinement and Generation Guidance</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown incredible potential in code generation tasks, and recent research in prompt engineering have enhanced LLMs' understanding of textual information. However, ensuring the accuracy of generated code often requires extensive testing and validation by programmers. While LLMs can typically generate code based on task descriptions, their accuracy remains limited, especially for complex tasks that require a deeper understanding of both the problem statement and the code generation process. This limitation is primarily due to the LLMs' need to simultaneously comprehend text and generate syntactically and semantically correct code, without having the capability to automatically refine the code. In real-world software development, programmers rarely produce flawless code in a single attempt based on the task description alone, they rely on iterative feedback and debugging to refine their programs. Inspired by this process, we introduce a novel architecture of LLM-based agents for code generation and automatic debugging: Refinement and Guidance Debugging (RGD). The RGD framework is a multi-LLM-based agent debugger that leverages three distinct LLM agents-Guide Agent, Debug Agent, and Feedback Agent. RGD decomposes the code generation task into multiple steps, ensuring a clearer workflow and enabling iterative code refinement based on self-reflection and feedback. Experimental results demonstrate that RGD exhibits remarkable code generation capabilities, achieving state-of-the-art performance with a 9.8% improvement on the HumanEval dataset and a 16.2% improvement on the MBPP dataset compared to the state-of-the-art approaches and traditional direct prompting approaches. We highlight the effectiveness of the RGD framework in enhancing LLMs' ability to generate and refine code autonomously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02425v1">LLM-Pilot: Characterize and Optimize Performance of your LLM Inference Services</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted to the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '24)
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are rapidly growing in popularity, LLM inference services must be able to serve requests from thousands of users while satisfying performance requirements. The performance of an LLM inference service is largely determined by the hardware onto which it is deployed, but understanding of which hardware will deliver on performance requirements remains challenging. In this work we present LLM-Pilot - a first-of-its-kind system for characterizing and predicting performance of LLM inference services. LLM-Pilot performs benchmarking of LLM inference services, under a realistic workload, across a variety of GPUs, and optimizes the service configuration for each considered GPU to maximize performance. Finally, using this characterization data, LLM-Pilot learns a predictive model, which can be used to recommend the most cost-effective hardware for a previously unseen LLM. Compared to existing methods, LLM-Pilot can deliver on performance requirements 33% more frequently, whilst reducing costs by 60% on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02406v1">ELLMA-T: an Embodied LLM-agent for Supporting English Language Learning in Social VR</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Many people struggle with learning a new language, with traditional tools falling short in providing contextualized learning tailored to each learner's needs. The recent development of large language models (LLMs) and embodied conversational agents (ECAs) in social virtual reality (VR) provide new opportunities to practice language learning in a contextualized and naturalistic way that takes into account the learner's language level and needs. To explore this opportunity, we developed ELLMA-T, an ECA that leverages an LLM (GPT-4) and situated learning framework for supporting learning English language in social VR (VRChat). Drawing on qualitative interviews (N=12), we reveal the potential of ELLMA-T to generate realistic, believable and context-specific role plays for agent-learner interaction in VR, and LLM's capability to provide initial language assessment and continuous feedback to learners. We provide five design implications for the future development of LLM-based language agents in social VR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14177v2">PathSeeker: Exploring LLM Security Vulnerabilities with a Reinforcement Learning-Based Jailbreak Approach</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 update the abstract and cite a new related work
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have gained widespread use, raising concerns about their security. Traditional jailbreak attacks, which often rely on the model internal information or have limitations when exploring the unsafe behavior of the victim model, limiting their reducing their general applicability. In this paper, we introduce PathSeeker, a novel black-box jailbreak method, which is inspired by the game of rats escaping a maze. We think that each LLM has its unique "security maze", and attackers attempt to find the exit learning from the received feedback and their accumulated experience to compromise the target LLM's security defences. Our approach leverages multi-agent reinforcement learning, where smaller models collaborate to guide the main LLM in performing mutation operations to achieve the attack objectives. By progressively modifying inputs based on the model's feedback, our system induces richer, harmful responses. During our manual attempts to perform jailbreak attacks, we found that the vocabulary of the response of the target model gradually became richer and eventually produced harmful responses. Based on the observation, we also introduce a reward mechanism that exploits the expansion of vocabulary richness in LLM responses to weaken security constraints. Our method outperforms five state-of-the-art attack techniques when tested across 13 commercial and open-source LLMs, achieving high attack success rates, especially in strongly aligned commercial models like GPT-4o-mini, Claude-3.5, and GLM-4-air with strong safety alignment. This study aims to improve the understanding of LLM security vulnerabilities and we hope that this sturdy can contribute to the development of more robust defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17419v2">Leave No Document Behind: Benchmarking Long-Context LLMs with Extended Multi-Doc QA</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 EMNLP 2024 Main. We release our code and data publicly at https://github.com/MozerWang/Loong
    </div>
    <details class="paper-abstract">
      Long-context modeling capabilities have garnered widespread attention, leading to the emergence of Large Language Models (LLMs) with ultra-context windows. Meanwhile, benchmarks for evaluating long-context LLMs are gradually catching up. However, existing benchmarks employ irrelevant noise texts to artificially extend the length of test cases, diverging from the real-world scenarios of long-context applications. To bridge this gap, we propose a novel long-context benchmark, Loong, aligning with realistic scenarios through extended multi-document question answering (QA). Unlike typical document QA, in Loong's test cases, each document is relevant to the final answer, ignoring any document will lead to the failure of the answer. Furthermore, Loong introduces four types of tasks with a range of context lengths: Spotlight Locating, Comparison, Clustering, and Chain of Reasoning, to facilitate a more realistic and comprehensive evaluation of long-context understanding. Extensive experiments indicate that existing long-context language models still exhibit considerable potential for enhancement. Retrieval augmented generation (RAG) achieves poor performance, demonstrating that Loong can reliably assess the model's long-context modeling capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00617v3">Iterative Nash Policy Optimization: Aligning LLMs with General Preferences via No-Regret Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) has achieved great success in aligning large language models (LLMs) with human preferences. Prevalent RLHF approaches are reward-based, following the Bradley-Terry (BT) model assumption, which may not fully capture the complexity of human preferences. In this paper, we explore RLHF under a general preference framework and approach it from a game-theoretic perspective. Specifically, we formulate the problem as a two-player game and propose a novel online algorithm, iterative Nash policy optimization (INPO). The key idea is to let the policy play against itself via no-regret learning, thereby approximating the Nash policy. Unlike previous methods, INPO bypasses the need for estimating the expected win rate for individual responses, which typically incurs high computational or annotation costs. Instead, we introduce a new loss objective that is directly minimized over a preference dataset. We provide theoretical analysis for our approach and demonstrate its effectiveness through experiments on various representative benchmarks. With an LLaMA-3-8B-based SFT model, INPO achieves a 42.6% length-controlled win rate on AlpacaEval 2.0 and a 37.8% win rate on Arena-Hard, showing substantial improvement over the state-of-the-art online RLHF algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02173v1">Efficiently Deploying LLMs with Controlled Risk</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Deploying large language models in production requires simultaneous attention to efficiency and risk control. Prior work has shown the possibility to cut costs while maintaining similar accuracy, but has neglected to focus on risk control. By contrast, here we present hierarchical chains with multi-level abstention (HCMA), which use model-intrinsic uncertainty to delegate queries along the LLM intelligence hierarchy, enabling training-free model switching based solely on black-box API calls. Our framework presents novel trade-offs between efficiency and risk. For example, deploying HCMA on MMLU cuts the error rate of Llama3 405B by 30% when the model is allowed to abstain on 20% of the queries. To calibrate HCMA for optimal performance, our approach uses data-efficient logistic regressions (based on a simple nonlinear feature transformation), which require only 50 or 100 labeled examples to achieve excellent calibration error (ECE), cutting ECE by 50% compared to naive Platt scaling. On free-form generation tasks, we find that chain-of-thought is ineffectual for selective prediction, whereas zero-shot prompting drives error to 0% on TruthfulQA at high abstention rates. As LLMs are increasingly deployed across computing environments with different capabilities (such as mobile, laptop, and cloud), our framework paves the way towards maintaining deployment efficiency while putting in place sharp risk controls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02165v1">A LLM-Powered Automatic Grading Framework with Human-Level Guidelines Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Open-ended short-answer questions (SAGs) have been widely recognized as a powerful tool for providing deeper insights into learners' responses in the context of learning analytics (LA). However, SAGs often present challenges in practice due to the high grading workload and concerns about inconsistent assessments. With recent advancements in natural language processing (NLP), automatic short-answer grading (ASAG) offers a promising solution to these challenges. Despite this, current ASAG algorithms are often limited in generalizability and tend to be tailored to specific questions. In this paper, we propose a unified multi-agent ASAG framework, GradeOpt, which leverages large language models (LLMs) as graders for SAGs. More importantly, GradeOpt incorporates two additional LLM-based agents - the reflector and the refiner - into the multi-agent system. This enables GradeOpt to automatically optimize the original grading guidelines by performing self-reflection on its errors. Through experiments on a challenging ASAG task, namely the grading of pedagogical content knowledge (PCK) and content knowledge (CK) questions, GradeOpt demonstrates superior performance in grading accuracy and behavior alignment with human graders compared to representative baselines. Finally, comprehensive ablation studies confirm the effectiveness of the individual components designed in GradeOpt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16253v3">LLMs Assist NLP Researchers: Critique Paper (Meta-)Reviewing</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Accepted by EMNLP 2024 main conference
    </div>
    <details class="paper-abstract">
      This work is motivated by two key trends. On one hand, large language models (LLMs) have shown remarkable versatility in various generative tasks such as writing, drawing, and question answering, significantly reducing the time required for many routine tasks. On the other hand, researchers, whose work is not only time-consuming but also highly expertise-demanding, face increasing challenges as they have to spend more time reading, writing, and reviewing papers. This raises the question: how can LLMs potentially assist researchers in alleviating their heavy workload? This study focuses on the topic of LLMs assist NLP Researchers, particularly examining the effectiveness of LLM in assisting paper (meta-)reviewing and its recognizability. To address this, we constructed the ReviewCritique dataset, which includes two types of information: (i) NLP papers (initial submissions rather than camera-ready) with both human-written and LLM-generated reviews, and (ii) each review comes with "deficiency" labels and corresponding explanations for individual segments, annotated by experts. Using ReviewCritique, this study explores two threads of research questions: (i) "LLMs as Reviewers", how do reviews generated by LLMs compare with those written by humans in terms of quality and distinguishability? (ii) "LLMs as Metareviewers", how effectively can LLMs identify potential issues, such as Deficient or unprofessional review segments, within individual paper reviews? To our knowledge, this is the first work to provide such a comprehensive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05467v3">Can Active Label Correction Improve LLM-based Modular AI Systems?</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 EMNLP (Main) 2024, 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Modular AI systems can be developed using LLM-prompts-based modules to minimize deployment time even for complex tasks. However, these systems do not always perform well and improving them using the data traces collected from a deployment remains an open challenge. The data traces contain LLM inputs and outputs, but the annotations from LLMs are noisy. We hypothesize that Active Label Correction (ALC) can be use on the collected data to train smaller task-specific improved models that can replace LLM-based modules. In this paper, we study the noise in three GPT-3.5-annotated datasets and their denoising with human feedback. We also propose a novel method ALC3 that iteratively applies three updates to the training dataset: auto-correction, correction using human feedback and filtering. Our results show that ALC3 can lead to oracle performance with feedback on 17-24% fewer examples than the number of noisy examples in the dataset across three different NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06904v2">Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 Preprint; First three authors contributed equally
    </div>
    <details class="paper-abstract">
      The Chain-of-Thought (CoT) paradigm has become a pivotal method for solving complex problems. However, its application to intricate, domain-specific tasks remains challenging, as large language models (LLMs) often struggle to accurately decompose these tasks and, even when decomposition is correct, fail to execute the subtasks effectively. This paper introduces the Re-TASK framework, a novel theoretical model that revisits LLM tasks from the perspectives of capability, skill, and knowledge, drawing on the principles of Bloom's Taxonomy and Knowledge Space Theory. While CoT offers a workflow perspective on tasks, the Re-TASK framework introduces a Chain-of-Learning view, illustrating how tasks and their corresponding subtasks depend on various capability items. Each capability item is further dissected into its constituent aspects of knowledge and skills. Our framework reveals that many CoT failures in domain-specific tasks stem from insufficient knowledge or inadequate skill adaptation. In response, we combine CoT with the Re-TASK framework and implement a carefully designed Re-TASK prompting strategy to improve task performance. Specifically, we identify core capability items linked to tasks and subtasks, then strengthen these capabilities through targeted knowledge injection and skill adaptation. We validate the Re-TASK framework on three datasets across the law, finance, and mathematics domains, achieving significant improvements over the baseline models. Notably, our approach yields a remarkable 44.42% improvement with the Yi-1.5-9B model and a 33.08% improvement with the Llama3-Chinese-8b on the legal dataset. These experimental results confirm the effectiveness of the Re-TASK framework, demonstrating substantial enhancements in both the performance and applicability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02108v1">ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2024-10-03
    </div>
    <details class="paper-abstract">
      Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.01026v1">Understanding the Human-LLM Dynamic: A Literature Survey of LLM Use in Programming Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-03
      | 💬 16 pages, 8 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming programming practices, offering significant capabilities for code generation activities. While researchers have explored the potential of LLMs in various domains, this paper focuses on their use in programming tasks, drawing insights from user studies that assess the impact of LLMs on programming tasks. We first examined the user interaction behaviors with LLMs observed in these studies, from the types of requests made to task completion strategies. Additionally, our analysis reveals both benefits and weaknesses of LLMs showing mixed effects on the human and task. Lastly, we looked into what factors from the human, LLM or the interaction of both, affect the human's enhancement as well as the task performance. Our findings highlight the variability in human-LLM interactions due to the non-deterministic nature of both parties (humans and LLMs), underscoring the need for a deeper understanding of these interaction patterns. We conclude by providing some practical suggestions for researchers as well as programmers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10490v2">Learning Dynamics of LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2024-10-02
    </div>
    <details class="paper-abstract">
      Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
    </details>
</div>
