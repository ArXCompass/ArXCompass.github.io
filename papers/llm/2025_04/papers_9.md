# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v1">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-04-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01700v1">Reasoning LLMs for User-Aware Multimodal Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Personalization in social robotics is critical for fostering effective human-robot interactions, yet systems often face the cold start problem, where initial user preferences or characteristics are unavailable. This paper proposes a novel framework called USER-LLM R1 for a user-aware conversational agent that addresses this challenge through dynamic user profiling and model initiation. Our approach integrates chain-of-thought (CoT) reasoning models to iteratively infer user preferences and vision-language models (VLMs) to initialize user profiles from multimodal inputs, enabling personalized interactions from the first encounter. Leveraging a Retrieval-Augmented Generation (RAG) architecture, the system dynamically refines user representations within an inherent CoT process, ensuring contextually relevant and adaptive responses. Evaluations on the ElderlyTech-VQA Bench demonstrate significant improvements in ROUGE-1 (+23.2%), ROUGE-2 (+0.6%), and ROUGE-L (+8%) F1 scores over state-of-the-art baselines, with ablation studies underscoring the impact of reasoning model size on performance. Human evaluations further validate the framework's efficacy, particularly for elderly users, where tailored responses enhance engagement and trust. Ethical considerations, including privacy preservation and bias mitigation, are rigorously discussed and addressed to ensure responsible deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01698v1">ToM-RL: Reinforcement Learning Unlocks Theory of Mind in Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Recent advancements in rule-based reinforcement learning (RL), applied during the post-training phase of large language models (LLMs), have significantly enhanced their capabilities in structured reasoning tasks such as mathematics and logical inference. However, the effectiveness of RL in social reasoning, particularly in Theory of Mind (ToM), the ability to infer others' mental states, remains largely unexplored. In this study, we demonstrate that RL methods effectively unlock ToM reasoning capabilities even in small-scale LLMs (0.5B to 7B parameters). Using a modest dataset comprising 3200 questions across diverse scenarios, our RL-trained 7B model achieves 84.50\% accuracy on the Hi-ToM benchmark, surpassing models like GPT-4o and DeepSeek-v3 despite significantly fewer parameters. While smaller models ($\leq$3B parameters) suffer from reasoning collapse, larger models (7B parameters) maintain stable performance through consistent belief tracking. Additionally, our RL-based models demonstrate robust generalization to higher-order, out-of-distribution ToM problems, novel textual presentations, and previously unseen datasets. These findings highlight RL's potential to enhance social cognitive reasoning, bridging the gap between structured problem-solving and nuanced social inference in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00595v2">Open-Qwen2VL: Compute-Efficient Pre-Training of Fully-Open Multimodal LLMs on Academic Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      The reproduction of state-of-the-art multimodal LLM pre-training faces barriers at every stage of the pipeline, including high-quality data filtering, multimodal data mixture strategies, sequence packing techniques, and training frameworks. We introduce Open-Qwen2VL, a fully open-source 2B-parameter Multimodal Large Language Model pre-trained efficiently on 29M image-text pairs using only 220 A100-40G GPU hours. Our approach employs low-to-high dynamic image resolution and multimodal sequence packing to significantly enhance pre-training efficiency. The training dataset was carefully curated using both MLLM-based filtering techniques (e.g., MLM-Filter) and conventional CLIP-based filtering methods, substantially improving data quality and training efficiency. The Open-Qwen2VL pre-training is conducted on academic level 8xA100-40G GPUs at UCSB on 5B packed multimodal tokens, which is 0.36% of 1.4T multimodal pre-training tokens of Qwen2-VL. The final instruction-tuned Open-Qwen2VL outperforms partially-open state-of-the-art MLLM Qwen2-VL-2B on various multimodal benchmarks of MMBench, SEEDBench, MMstar, and MathVista, indicating the remarkable training efficiency of Open-Qwen2VL. We open-source all aspects of our work, including compute-efficient and data-efficient training details, data filtering methods, sequence packing scripts, pre-training data in WebDataset format, FSDP-based training codebase, and both base and instruction-tuned model checkpoints. We redefine "fully open" for multimodal LLMs as the complete release of: 1) the training codebase, 2) detailed data filtering techniques, and 3) all pre-training and supervised fine-tuning data used to develop the model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01602v1">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted by WWW 2025 Industry Track
    </div>
    <details class="paper-abstract">
      In modern online streaming platforms, the comments section plays a critical role in enhancing the overall user experience. Understanding user behavior within the comments section is essential for comprehensive user interest modeling. A key factor of user engagement is staytime, which refers to the amount of time that users browse and post comments. Existing watchtime prediction methods struggle to adapt to staytime prediction, overlooking interactions with individual comments and their interrelation. In this paper, we present a micro-video recommendation dataset with video comments (named as KuaiComt) which is collected from Kuaishou platform. correspondingly, we propose a practical framework for comment staytime prediction with LLM-enhanced Comment Understanding (LCU). Our framework leverages the strong text comprehension capabilities of large language models (LLMs) to understand textual information of comments, while also incorporating fine-grained comment ranking signals as auxiliary tasks. The framework is two-staged: first, the LLM is fine-tuned using domain-specific tasks to bridge the video and the comments; second, we incorporate the LLM outputs into the prediction model and design two comment ranking auxiliary tasks to better understand user preference. Extensive offline experiments demonstrate the effectiveness of our framework, showing significant improvements on the task of comment staytime prediction. Additionally, online A/B testing further validates the practical benefits on industrial scenario. Our dataset KuaiComt (https://github.com/lyingCS/KuaiComt.github.io) and code for LCU (https://github.com/lyingCS/LCU) are fully released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01588v1">Building Knowledge from Interactions: An LLM-Based Architecture for Adaptive Tutoring and Social Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
    </div>
    <details class="paper-abstract">
      Integrating robotics into everyday scenarios like tutoring or physical training requires robots capable of adaptive, socially engaging, and goal-oriented interactions. While Large Language Models show promise in human-like communication, their standalone use is hindered by memory constraints and contextual incoherence. This work presents a multimodal, cognitively inspired framework that enhances LLM-based autonomous decision-making in social and task-oriented Human-Robot Interaction. Specifically, we develop an LLM-based agent for a robot trainer, balancing social conversation with task guidance and goal-driven motivation. To further enhance autonomy and personalization, we introduce a memory system for selecting, storing and retrieving experiences, facilitating generalized reasoning based on knowledge built across different interactions. A preliminary HRI user study and offline experiments with a synthetic dataset validate our approach, demonstrating the system's ability to manage complex interactions, autonomously drive training tasks, and build and retrieve contextual memories, advancing socially intelligent robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01542v1">Register Always Matters: Analysis of LLM Pretraining Data Through the Lens of Language Variation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Pretraining data curation is a cornerstone in Large Language Model (LLM) development, leading to growing research on quality filtering of large web corpora. From statistical quality flags to LLM-based labeling systems, datasets are divided into categories, frequently reducing to a binary: those passing the filters deemed as valuable examples, others discarded as useless or detrimental. However, a more detailed understanding of the contribution of different kinds of texts to model performance is still largely lacking. In this article, we present the first study utilizing registers (also known as genres) - a widely used standard in corpus linguistics to model linguistic variation - to curate pretraining datasets and investigate the effect of register on the performance of LLMs. We perform comparative studies by training models with register classified data and evaluating them using standard benchmarks, and show that the register of pretraining data substantially affects model performance. We uncover surprising relationships between the pretraining material and the resulting models: using the News register results in subpar performance, and on the contrary, including the Opinion class, covering texts such as reviews and opinion blogs, is highly beneficial. While a model trained on the entire unfiltered dataset outperforms those trained on datasets limited to a single register, combining well-performing registers like How-to-Instructions, Informational Description, and Opinion leads to major improvements. Furthermore, analysis of individual benchmark results reveals key differences in the strengths and drawbacks of specific register classes as pretraining data. These findings show that register is an important explainer of model variation and can facilitate more deliberate future data selection practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00762v2">Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scales Test-Time Compute</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11313v2">An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Be accepeted as NAACL2025 Findings
    </div>
    <details class="paper-abstract">
      Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.08460v3">Is Your LLM Outdated? A Deep Look at Temporal Generalization</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 NAACL 2025 Oral
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to the development of benchmarks that consider temporal dynamics, however, there remains a gap in understanding how well these models can generalize across temporal contexts due to the inherent dynamic nature of language and information. This paper introduces the concept of temporal generalization in LLMs, including bias in past and future generalizations. Then we introduce FreshBench, a new evaluation framework that employs fresh text and event prediction for assessing LLMs' temporal adaptability, ensuring the evaluation process free from data leakage and subjective bias. The experiment shows significant temporal biases and a decline in performance over time. Our findings reveal that powerful models, while initially superior, tend to decline more rapidly in future generalization. Additionally, powerful open-source models demonstrate better long-term adaptability compared to their closed-source counterparts. Our code is available at https://github.com/FreedomIntelligence/FreshBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05939v2">Direct Preference Optimization for LLM-Enhanced Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 This paper has been accepted to ICME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited remarkable performance across a wide range of domains, motivating research into their potential for recommendation systems. Early efforts have leveraged LLMs' rich knowledge and strong generalization capabilities via in-context learning, where recommendation tasks are framed as prompts. However, LLM performance in recommendation scenarios remains limited due to the mismatch between their pretraining objectives and recommendation tasks, as well as the lack of recommendation-specific data during pretraining. To address these challenges, we propose DPO4Rec, a novel framework that integrates Direct Preference Optimization (DPO) into LLM-enhanced recommendation systems. First, we prompt the LLM to infer user preferences from historical interactions, which are then used to augment traditional ID-based sequential recommendation models. Next, we train a reward model based on knowledge-augmented recommendation architectures to assess the quality of LLM-generated reasoning. Using this, we select the highest- and lowest-ranked responses from N samples to construct a dataset for LLM fine-tuning. Finally, we apply a structure alignment strategy via DPO to align the LLM's outputs with desirable recommendation behavior. Extensive experiments show that DPO4Rec significantly improves re-ranking performance over strong baselines, demonstrating enhanced instruction-following capabilities of LLMs in recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16863v2">Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 CVPR 2025
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) are the natural extension of large language models to handle multimodal inputs, combining text and image data. They have recently garnered attention due to their capability to address complex tasks involving both modalities. However, their effectiveness is limited to the knowledge acquired during training, which restricts their practical utility. In this work, we introduce a novel method to enhance the adaptability of MLLMs by integrating external knowledge sources. Our proposed model, Reflective LLaVA (ReflectiVA), utilizes reflective tokens to dynamically determine the need for external knowledge and predict the relevance of information retrieved from an external database. Tokens are trained following a two-stage two-model training recipe. This ultimately enables the MLLM to manage external knowledge while preserving fluency and performance on tasks where external knowledge is not needed. Through our experiments, we demonstrate the efficacy of ReflectiVA for knowledge-based visual question answering, highlighting its superior performance compared to existing methods. Source code and trained models are publicly available at https://aimagelab.github.io/ReflectiVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01369v1">LITE: LLM-Impelled efficient Taxonomy Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This paper presents LITE, an LLM-based evaluation method designed for efficient and flexible assessment of taxonomy quality. To address challenges in large-scale taxonomy evaluation, such as efficiency, fairness, and consistency, LITE adopts a top-down hierarchical evaluation strategy, breaking down the taxonomy into manageable substructures and ensuring result reliability through cross-validation and standardized input formats. LITE also introduces a penalty mechanism to handle extreme cases and provides both quantitative performance analysis and qualitative insights by integrating evaluation metrics closely aligned with task objectives. Experimental results show that LITE demonstrates high reliability in complex evaluation tasks, effectively identifying semantic errors, logical contradictions, and structural flaws in taxonomies, while offering directions for improvement. Code is available at https://github.com/Zhang-l-i-n/TAXONOMY_DETECT .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21393v2">An evaluation of LLMs and Google Translate for translation of selected Indian languages via sentiment and semantic analyses</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have been prominent for language translation, including low-resource languages. There has been limited study about the assessment of the quality of translations generated by LLMs, including Gemini, GPT and Google Translate. In this study, we address this limitation by using semantic and sentiment analysis of selected LLMs for Indian languages, including Sanskrit, Telugu and Hindi. We select prominent texts that have been well translated by experts and use LLMs to generate their translations to English, and then we provide a comparison with selected expert (human) translations. Our findings suggest that while LLMs have made significant progress in translation accuracy, challenges remain in preserving sentiment and semantic integrity, especially in figurative and philosophical contexts. The sentiment analysis revealed that GPT-4o and GPT-3.5 are better at preserving the sentiments for the Bhagavad Gita (Sanskrit-English) translations when compared to Google Translate. We observed a similar trend for the case of Tamas (Hindi-English) and Maha P (Telugu-English) translations. GPT-4o performs similarly to GPT-3.5 in the translation in terms of sentiments for the three languages. We found that LLMs are generally better at translation for capturing sentiments when compared to Google Translate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01304v1">Real-time Ad retrieval via LLM-generative Commercial Intention for Sponsored Search Advertising</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 13pages,5 figures
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) with retrieval systems has shown promising potential in retrieving documents (docs) or advertisements (ads) for a given query. Existing LLM-based retrieval methods generate numeric or content-based DocIDs to retrieve docs/ads. However, the one-to-few mapping between numeric IDs and docs, along with the time-consuming content extraction, leads to semantic inefficiency and limits scalability in large-scale corpora. In this paper, we propose the Real-time Ad REtrieval (RARE) framework, which leverages LLM-generated text called Commercial Intentions (CIs) as an intermediate semantic representation to directly retrieve ads for queries in real-time. These CIs are generated by a customized LLM injected with commercial knowledge, enhancing its domain relevance. Each CI corresponds to multiple ads, yielding a lightweight and scalable set of CIs. RARE has been implemented in a real-world online system, handling daily search volumes in the hundreds of millions. The online implementation has yielded significant benefits: a 5.04% increase in consumption, a 6.37% rise in Gross Merchandise Volume (GMV), a 1.28% enhancement in click-through rate (CTR) and a 5.29% increase in shallow conversions. Extensive offline experiments show RARE's superiority over ten competitive baselines in four major categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01296v1">ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We present ThinkPrune, a simple yet effective method for pruning the thinking length for long-thinking LLMs, which has been found to often produce inefficient and redundant thinking processes. Existing preliminary explorations of reducing thinking length primarily focus on forcing the thinking process to early exit, rather than adapting the LLM to optimize and consolidate the thinking process, and therefore the length-performance tradeoff observed so far is sub-optimal. To fill this gap, ThinkPrune offers a simple solution that continuously trains the long-thinking LLMs via reinforcement learning (RL) with an added token limit, beyond which any unfinished thoughts and answers will be discarded, resulting in a zero reward. To further preserve model performance, we introduce an iterative length pruning approach, where multiple rounds of RL are conducted, each with an increasingly more stringent token limit. We observed that ThinkPrune results in a remarkable performance-length tradeoff -- on the AIME24 dataset, the reasoning length of DeepSeek-R1-Distill-Qwen-1.5B can be reduced by half with only 2% drop in performance. We also observed that after pruning, the LLMs can bypass unnecessary steps while keeping the core reasoning process complete. Code is available at https://github.com/UCSB-NLP-Chang/ThinkPrune.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01294v1">Extracting Formal Specifications from Documents Using LLMs for Automated Testing</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Automated testing plays a crucial role in ensuring software security. It heavily relies on formal specifications to validate the correctness of the system behavior. However, the main approach to defining these formal specifications is through manual analysis of software documents, which requires a significant amount of engineering effort from experienced researchers and engineers. Meanwhile, system update further increases the human labor cost to maintain a corresponding formal specification, making the manual analysis approach a time-consuming and error-prone task. Recent advances in Large Language Models (LLMs) have demonstrated promising capabilities in natural language understanding. Yet, the feasibility of using LLMs to automate the extraction of formal specifications from software documents remains unexplored. We conduct an empirical study by constructing a comprehensive dataset comprising 603 specifications from 37 documents across three representative open-source software. We then evaluate the most recent LLMs' capabilities in extracting formal specifications from documents in an end-to-end fashion, including GPT-4o, Claude, and Llama. Our study demonstrates the application of LLMs in formal specification extraction tasks while identifying two major limitations: specification oversimplification and specification fabrication. We attribute these deficiencies to the LLMs' inherent limitations in processing and expressive capabilities, as well as their tendency to fabricate fictional information. Inspired by human cognitive processes, we propose a two-stage method, annotation-then-conversion, to address these challenges. Our method demonstrates significant improvements over the end-to-end method, with a 29.2% increase in the number of correctly extracted specifications and a 14.0% improvement in average accuracy. In particular, our best-performing LLM achieves an accuracy of 71.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v2">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01282v1">Prompt-Reverse Inconsistency: LLM Self-Inconsistency Beyond Generative Randomness and Prompt Paraphrasing</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      While the inconsistency of LLMs is not a novel topic, prior research has predominantly addressed two types of generative inconsistencies: i) Randomness Inconsistency: running the same LLM multiple trials, yielding varying responses; ii) Paraphrase Inconsistency: paraphrased prompts result in different responses from the same LLM. Randomness Inconsistency arises from the inherent randomness due to stochastic sampling in generative models, while Paraphrase Inconsistency is a consequence of the language modeling objectives, where paraphrased prompts alter the distribution of vocabulary logits. This research discovers Prompt-Reverse Inconsistency (PRIN), a new form of LLM self-inconsistency: given a question and a couple of LLM-generated answer candidates, the LLM often has conflicting responses when prompted "Which are correct answers?" and "Which are incorrect answers?". PRIN poses a big concern as it undermines the credibility of LLM-as-a-judge, and suggests a challenge for LLMs to adhere to basic logical rules. We conduct a series of experiments to investigate PRIN, examining the extent of PRIN across different LLMs, methods to mitigate it, potential applications, and its relationship with Randomness Inconsistency and Paraphrase Inconsistency. As the first study to explore PRIN, our findings offer valuable insights into the inner workings of LLMs and contribute to advancing trustworthy AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01259v1">Facilitating Instructors-LLM Collaboration for Problem Design in Introductory Programming Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      Advancements in Large Language Models (LLMs), such as ChatGPT, offer significant opportunities to enhance instructional support in introductory programming courses. While extensive research has explored the effectiveness of LLMs in supporting student learning, limited studies have examined how these models can assist instructors in designing instructional activities. This work investigates how instructors' expertise in effective activity design can be integrated with LLMs' ability to generate novel and targeted programming problems, facilitating more effective activity creation for programming classrooms. To achieve this, we employ a participatory design approach to develop an instructor-authoring tool that incorporates LLM support, fostering collaboration between instructors and AI in generating programming exercises. This tool also allows instructors to specify common student mistakes and misconceptions, which informs the adaptive feedback generation process. We conduct case studies with three instructors, analyzing how they use our system to design programming problems for their introductory courses. Through these case studies, we assess instructors' perceptions of the usefulness and limitations of LLMs in authoring problem statements for instructional purposes. Additionally, we compare the efficiency, quality, effectiveness, and coverage of designed activities when instructors create problems with and without structured LLM prompting guidelines. Our findings provide insights into the potential of LLMs in enhancing instructor workflows and improving programming education and provide guidelines for designing effective AI-assisted problem-authoring interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02891v1">Automated Survey Collection with LLM-based Conversational Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Objective: Traditional phone-based surveys are among the most accessible and widely used methods to collect biomedical and healthcare data, however, they are often costly, labor intensive, and difficult to scale effectively. To overcome these limitations, we propose an end-to-end survey collection framework driven by conversational Large Language Models (LLMs). Materials and Methods: Our framework consists of a researcher responsible for designing the survey and recruiting participants, a conversational phone agent powered by an LLM that calls participants and administers the survey, a second LLM (GPT-4o) that analyzes the conversation transcripts generated during the surveys, and a database for storing and organizing the results. To test our framework, we recruited 8 participants consisting of 5 native and 3 non-native english speakers and administered 40 surveys. We evaluated the correctness of LLM-generated conversation transcripts, accuracy of survey responses inferred by GPT-4o and overall participant experience. Results: Survey responses were successfully extracted by GPT-4o from conversation transcripts with an average accuracy of 98% despite transcripts exhibiting an average per-line word error rate of 7.7%. While participants noted occasional errors made by the conversational LLM agent, they reported that the agent effectively conveyed the purpose of the survey, demonstrated good comprehension, and maintained an engaging interaction. Conclusions: Our study highlights the potential of LLM agents in conducting and analyzing phone surveys for healthcare applications. By reducing the workload on human interviewers and offering a scalable solution, this approach paves the way for real-world, end-to-end AI-powered phone survey collection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02890v1">Scaling Test-time Compute for Low-resource Languages: Multilingual Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Recent advances in test-time compute scaling have enabled Large Language Models (LLMs) to tackle deep reasoning tasks by generating a chain-of-thought (CoT) that includes trial and error, backtracking, and intermediate reasoning steps before producing the final answer. However, these techniques have been applied predominantly to popular languages, such as English, leaving reasoning in low-resource languages underexplored and misaligned. In this work, we investigate the multilingual mechanism by which LLMs internally operate in a latent space biased toward their inherently dominant language. To leverage this phenomenon for low-resource languages, we train models to generate the CoT in English while outputting the final response in the target language, given input in the low-resource language. Our experiments demonstrate that this approach, named English-Pivoted CoT Training, outperforms other baselines, including training to generate both the CoT and the final response solely in the target language, with up to 28.33% improvement. Further analysis provides novel insights into the relationships between reasoning and multilinguality of LLMs, prompting for better approaches in developing multilingual large reasoning models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02878v1">Exploring the Capabilities of LLMs for IMU-based Fine-grained Human Activity Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted to The 2nd International Workshop on Foundation Models for Cyber-Physical Systems & Internet of Things (FMSys 2025)
    </div>
    <details class="paper-abstract">
      Human activity recognition (HAR) using inertial measurement units (IMUs) increasingly leverages large language models (LLMs), yet existing approaches focus on coarse activities like walking or running. Our preliminary study indicates that pretrained LLMs fail catastrophically on fine-grained HAR tasks such as air-written letter recognition, achieving only near-random guessing accuracy. In this work, we first bridge this gap for flat-surface writing scenarios: by fine-tuning LLMs with a self-collected dataset and few-shot learning, we achieved up to a 129x improvement on 2D data. To extend this to 3D scenarios, we designed an encoder-based pipeline that maps 3D data into 2D equivalents, preserving the spatiotemporal information for robust letter prediction. Our end-to-end pipeline achieves 78% accuracy on word recognition with up to 5 letters in mid-air writing scenarios, establishing LLMs as viable tools for fine-grained HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03767v1">MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment. The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/leidosinc/McpSafetyScanner
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16843v4">Do LLM Agents Have Regret? A Case Study in Online Learning and Games</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Camera ready version of ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been increasingly employed for (interactive) decision-making, via the development of LLM-based autonomous agents. Despite their emerging successes, the performance of LLM agents in decision-making has not been fully investigated through quantitative metrics, especially in the multi-agent setting when they interact with each other, a typical scenario in real-world LLM-agent applications. To better understand the limits of LLM agents in these interactive environments, we propose to study their interactions in benchmark decision-making settings in online learning and game theory, through the performance metric of \emph{regret}. We first empirically study the {no-regret} behaviors of LLMs in canonical (non-stationary) online learning problems, as well as the emergence of equilibria when LLM agents interact through playing repeated games. We then provide some theoretical insights into the no-regret behaviors of LLM agents, under certain assumptions on the supervised pre-training and the rationality model of human decision-makers who generate the data. Notably, we also identify (simple) cases where advanced LLMs such as GPT-4 fail to be no-regret. To promote the no-regret behaviors, we propose a novel \emph{unsupervised} training loss of \emph{regret-loss}, which, in contrast to the supervised pre-training loss, does not require the labels of (optimal) actions. We then establish the statistical guarantee of generalization bound for regret-loss minimization, followed by the optimization guarantee that minimizing such a loss may automatically lead to known no-regret learning algorithms. Our further experiments demonstrate the effectiveness of our regret-loss, especially in addressing the above ``regrettable'' cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21983v2">Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI Coevolution
    </div>
    <details class="paper-abstract">
      As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02165v1">Responsible Innovation: A Strategic Framework for Financial LLM Integration</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 27 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Financial institutions of all sizes are increasingly adopting Large Language Models (LLMs) to enhance credit assessments, deliver personalized client advisory services, and automate various language-intensive processes. However, effectively deploying LLMs requires careful management of stringent data governance requirements, heightened demands for interpretability, ethical responsibilities, and rapidly evolving regulatory landscapes. To address these challenges, we introduce a structured six-decision framework specifically designed for the financial sector, guiding organizations systematically from initial feasibility assessments to final deployment strategies. The framework encourages institutions to: (1) evaluate whether an advanced LLM is necessary at all, (2) formalize robust data governance and privacy safeguards, (3) establish targeted risk management mechanisms, (4) integrate ethical considerations early in the development process, (5) justify the initiative's return on investment (ROI) and strategic value, and only then (6) choose the optimal implementation pathway -- open-source versus proprietary, or in-house versus vendor-supported -- aligned with regulatory requirements and operational realities. By linking strategic considerations with practical steps such as pilot testing, maintaining comprehensive audit trails, and conducting ongoing compliance evaluations, this decision framework offers a structured roadmap for responsibly leveraging LLMs. Rather than acting as a rigid, one-size-fits-all solution, it shows how advanced language models can be thoughtfully integrated into existing workflows -- balancing innovation with accountability to uphold stakeholder trust and regulatory integrity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22296v4">Generalists vs. Specialists: Evaluating LLMs on Highly-Constrained Biophysical Sequence Optimization Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Supercedes arXiv:2407.00236v1. arXiv admin note: text overlap with arXiv:2407.00236
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown promise in biomolecule optimization problems, they incur heavy computational costs and struggle to satisfy precise constraints. On the other hand, specialized solvers like LaMBO-2 offer efficiency and fine-grained control but require more domain expertise. Comparing these approaches is challenging due to expensive laboratory validation and inadequate synthetic benchmarks. We address this by introducing Ehrlich functions, a synthetic test suite that captures the geometric structure of biophysical sequence optimization problems. With prompting alone, off-the-shelf LLMs struggle to optimize Ehrlich functions. In response, we propose LLOME (Language Model Optimization with Margin Expectation), a bilevel optimization routine for online black-box optimization. When combined with a novel preference learning loss, we find LLOME can not only learn to solve some Ehrlich functions, but can even outperform LaMBO-2 on moderately difficult Ehrlich variants. However, LLOME is comparable to LaMBO-2 on very easy or difficult variants, exhibits some likelihood-reward miscalibration, and struggles without explicit rewards. Our results indicate LLMs can provide significant benefits in some cases, but specialized solvers are still competitive and incur less overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12386v2">Interpretable LLM-based Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 10 pages, 2 figures and 9 tables in the main text
    </div>
    <details class="paper-abstract">
      Interpretability for Table Question Answering (Table QA) is critical, particularly in high-stakes industries like finance or healthcare. Although recent approaches using Large Language Models (LLMs) have significantly improved Table QA performance, their explanations for how the answers are generated are ambiguous. To fill this gap, we introduce Plan-of-SQLs (POS), an interpretable Table QA approach designed to improve users' understanding of model decision-making. Through qualitative and quantitative evaluations with human and LLM judges, we show that: First, POS is the highest-quality explanation method, helps human users understand model behaviors, and facilitates model prediction verification. Second, when evaluated on popular and standard Table QA datasets (TabFact, WikiTQ, and FetaQA), POS achieves QA accuracy that is competitive with or superior to existing methods, while also offering greater efficiency-requiring significantly fewer LLM calls and table database queries-and robust performance on large-sized tables. Finally, we observe high agreement (up to 90%) between LLMs and human users when making decisions based on the same explanations, suggesting that LLMs could serve as an effective proxy for humans in evaluating explanations. This finding enables faster, more affordable evaluation of AI explanations-possibly accelerating trustworthy AI research while maintaining reliable judgments on interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02141v1">On Simulation-Guided LLM-based Code Generation for Safe Autonomous Driving Software</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Accepted in the 29th International Conference on Evaluation and Assessment in Software Engineering (EASE)
    </div>
    <details class="paper-abstract">
      Automated Driving System (ADS) is a safety-critical software system responsible for the interpretation of the vehicle's environment and making decisions accordingly. The unbounded complexity of the driving context, including unforeseeable events, necessitate continuous improvement, often achieved through iterative DevOps processes. However, DevOps processes are themselves complex, making these improvements both time- and resource-intensive. Automation in code generation for ADS using Large Language Models (LLM) is one potential approach to address this challenge. Nevertheless, the development of ADS requires rigorous processes to verify, validate, assess, and qualify the code before it can be deployed in the vehicle and used. In this study, we developed and evaluated a prototype for automatic code generation and assessment using a designed pipeline of a LLM-based agent, simulation model, and rule-based feedback generator in an industrial setup. The LLM-generated code is evaluated automatically in a simulation model against multiple critical traffic scenarios, and an assessment report is provided as feedback to the LLM for modification or bug fixing. We report about the experimental results of the prototype employing Codellama:34b, DeepSeek (r1:32b and Coder:33b), CodeGemma:7b, Mistral:7b, and GPT4 for Adaptive Cruise Control (ACC) and Unsupervised Collision Avoidance by Evasive Manoeuvre (CAEM). We finally assessed the tool with 11 experts at two Original Equipment Manufacturers (OEMs) by conducting an interview study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17961v2">NormTab: Improving Symbolic Reasoning in LLMs Through Tabular Data Normalization</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 EMNLP 2024 (Findings)
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated remarkable capabilities in parsing textual data and generating code. However, their performance in tasks involving tabular data, especially those requiring symbolic reasoning, faces challenges due to the structural variance and inconsistency in table cell values often found in web tables. In this paper, we introduce NormTab, a novel framework aimed at enhancing the symbolic reasoning performance of LLMs by normalizing web tables. We study table normalization as a stand-alone, one-time preprocessing step using LLMs to support symbolic reasoning on tabular data. Our experimental evaluation, conducted on challenging web table datasets such as WikiTableQuestion and TabFact, demonstrates that leveraging NormTab significantly improves symbolic reasoning performance, showcasing the importance and effectiveness of web table normalization for enhancing LLM-based symbolic reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02119v1">Efficient Model Selection for Time Series Forecasting via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 16 pages, 3 Figures
    </div>
    <details class="paper-abstract">
      Model selection is a critical step in time series forecasting, traditionally requiring extensive performance evaluations across various datasets. Meta-learning approaches aim to automate this process, but they typically depend on pre-constructed performance matrices, which are costly to build. In this work, we propose to leverage Large Language Models (LLMs) as a lightweight alternative for model selection. Our method eliminates the need for explicit performance matrices by utilizing the inherent knowledge and reasoning capabilities of LLMs. Through extensive experiments with LLaMA, GPT and Gemini, we demonstrate that our approach outperforms traditional meta-learning techniques and heuristic baselines, while significantly reducing computational overhead. These findings underscore the potential of LLMs in efficient model selection for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02118v1">LLMPi: Optimizing LLMs for High-Throughput on Raspberry Pi</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on resource-constrained edge devices like the Raspberry Pi presents challenges in computational efficiency, power consumption, and response latency. This paper explores quantization-based optimization techniques to enable high-throughput, energy-efficient execution of LLMs on low-power embedded systems. Our approach leverages k-quantization, a Post-Training Quantization (PTQ) method designed for different bit-widths, enabling efficient 2-bit, 4-bit, 6-bit, and 8-bit weight quantization. Additionally, we employ ternary quantization using Quantization-Aware Training (QAT) for BitNet models, allowing for more effective adaptation to lower-bit representations while preserving accuracy. Our findings highlight the potential of quantized LLMs for real-time conversational AI on edge devices, paving the way for low-power, high-efficiency AI deployment in mobile and embedded applications. This study demonstrates that aggressive quantization strategies can significantly reduce energy consumption while maintaining inference quality, making LLMs practical for resource-limited environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02111v1">Exploring LLM Reasoning Through Controlled Prompt Variations</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This study investigates the reasoning robustness of large language models (LLMs) on mathematical problem-solving tasks under systematically introduced input perturbations. Using the GSM8K dataset as a controlled testbed, we evaluate how well state-of-the-art models maintain logical consistency and correctness when confronted with four categories of prompt perturbations: irrelevant context, pathological instructions, factually relevant but non-essential context, and a combination of the latter two. Our experiments, conducted on thirteen open-source and closed-source LLMs, reveal that introducing irrelevant context within the model's context window significantly degrades performance, suggesting that distinguishing essential from extraneous details remains a pressing challenge. Surprisingly, performance regressions are relatively insensitive to the complexity of the reasoning task, as measured by the number of steps required, and are not strictly correlated with model size. Moreover, we observe that certain perturbations inadvertently trigger chain-of-thought-like reasoning behaviors, even without explicit prompting. Our findings highlight critical vulnerabilities in current LLMs and underscore the need for improved robustness against noisy, misleading, and contextually dense inputs, paving the way for more resilient and reliable reasoning in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02107v1">TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Code available at: https://github.com/apple/ml-tic-lm
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v2">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02094v1">FlowDistill: Scalable Traffic Flow Prediction via Distillation from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Accurate traffic flow prediction is vital for optimizing urban mobility, yet it remains difficult in many cities due to complex spatio-temporal dependencies and limited high-quality data. While deep graph-based models demonstrate strong predictive power, their performance often comes at the cost of high computational overhead and substantial training data requirements, making them impractical for deployment in resource-constrained or data-scarce environments. We propose the FlowDistill, a lightweight and scalable traffic prediction framework based on knowledge distillation from large language models (LLMs). In this teacher-student setup, a fine-tuned LLM guides a compact multi-layer perceptron (MLP) student model using a novel combination of the information bottleneck principle and teacher-bounded regression loss, ensuring the distilled model retains only essential and transferable knowledge. Spatial and temporal correlations are explicitly encoded to enhance the model's generalization across diverse urban settings. Despite its simplicity, FlowDistill consistently outperforms state-of-the-art models in prediction accuracy while requiring significantly less training data, and achieving lower memory usage and inference latency, highlighting its efficiency and suitability for real-world, scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12491v2">Insights from the Inverse: Reconstructing LLM Training Goals Through Inverse Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained with Reinforcement Learning from Human Feedback (RLHF) have demonstrated remarkable capabilities, but their underlying reward functions and decision-making processes remain opaque. This paper introduces a novel approach to interpreting LLMs by applying inverse reinforcement learning (IRL) to recover their implicit reward functions. We conduct experiments on toxicity-aligned LLMs of varying sizes, extracting reward models that achieve up to 85\% accuracy in predicting human preferences. Our analysis reveals key insights into the non-identifiability of reward functions, the relationship between model size and interpretability, and potential pitfalls in the RLHF process. We demonstrate that IRL-derived reward models can be used to fine-tune new LLMs, resulting in comparable or improved performance on toxicity benchmarks. This work provides a new lens for understanding and improving LLM alignment, with implications for the responsible development and deployment of these powerful systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02080v1">Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content. In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety. Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance model robustness. Our study evaluates both open-source models (e.g., LLaMA and Mistral) and closed-source systems (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v4">Smart Routing: Cost-Effective Multi-LLM Serving for Multi-Core AIOS</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. For example, those simple queries can be safely handled by small, fast and cheap LLMs, while those complex and difficult queries need to be handled by large, slow, and expensive LLMs. This paper addresses this challenge by proposing an efficient capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset for sample-wise response quality and costs collected by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS, and the proposed smart routing mechanism has been integrated into AIOS, the AI Agent Operating System, at https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02074v1">Trapped by Expectations: Functional Fixedness in LLM-Enabled Chat Search</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Functional fixedness, a cognitive bias that restricts users' interactions with a new system or tool to expected or familiar ways, limits the full potential of Large Language Model (LLM)-enabled chat search, especially in complex and exploratory tasks. To investigate its impact, we conducted a crowdsourcing study with 450 participants, each completing one of six decision-making tasks spanning public safety, diet and health management, sustainability, and AI ethics. Participants engaged in a multi-prompt conversation with ChatGPT to address the task, allowing us to compare pre-chat intent-based expectations with observed interactions. We found that: 1) Several aspects of pre-chat expectations are closely associated with users' prior experiences with ChatGPT, search engines, and virtual assistants; 2) Prior system experience shapes language use and prompting behavior. Frequent ChatGPT users reduced deictic terms and hedge words and frequently adjusted prompts. Users with rich search experience maintained structured, less-conversational queries with minimal modifications. Users of virtual assistants favored directive, command-like prompts, reinforcing functional fixedness; 3) When the system failed to meet expectations, participants generated more detailed prompts with increased linguistic diversity, reflecting adaptive shifts. These findings suggest that while preconceived expectations constrain early interactions, unmet expectations can motivate behavioral adaptation. With appropriate system support, this may promote broader exploration of LLM capabilities. This work also introduces a typology for user intents in chat search and highlights the importance of mitigating functional fixedness to support more creative and analytical use of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02051v1">Self-Resource Allocation in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      With the development of LLMs as agents, there is a growing interest in connecting multiple agents into multi-agent systems to solve tasks concurrently, focusing on their role in task assignment and coordination. This paper explores how LLMs can effectively allocate computational tasks among multiple agents, considering factors such as cost, efficiency, and performance. In this work, we address key questions, including the effectiveness of LLMs as orchestrators and planners, comparing their effectiveness in task assignment and coordination. Our experiments demonstrate that LLMs can achieve high validity and accuracy in resource allocation tasks. We find that the planner method outperforms the orchestrator method in handling concurrent actions, resulting in improved efficiency and better utilization of agents. Additionally, we show that providing explicit information about worker capabilities enhances the allocation strategies of planners, particularly when dealing with suboptimal workers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01951v1">The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01911v1">Advancing AI-Scientist Understanding: Making LLM Think Like a Physicist with Interpretable Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are playing an expanding role in physics research by enhancing reasoning, symbolic manipulation, and numerical computation. However, ensuring the reliability and interpretability of their outputs remains a significant challenge. In our framework, we conceptualize the collaboration between AI and human scientists as a dynamic interplay among three modules: the reasoning module, the interpretation module, and the AI-scientist interaction module. Recognizing that effective physics reasoning demands rigorous logical consistency, quantitative precision, and deep integration with established theoretical models, we introduce the interpretation module to improve the understanding of AI-generated outputs, which is not previously explored in the literature. This module comprises multiple specialized agents, including summarizers, model builders, UI builders, and testers, which collaboratively structure LLM outputs within a physically grounded framework, by constructing a more interpretable science model. A case study demonstrates that our approach enhances transparency, facilitates validation, and strengthens AI-augmented reasoning in scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01903v1">STAR-1: Safer Alignment of Reasoning LLMs with 1K Data</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      This paper introduces STAR-1, a high-quality, just-1k-scale safety dataset specifically designed for large reasoning models (LRMs) like DeepSeek-R1. Built on three core principles -- diversity, deliberative reasoning, and rigorous filtering -- STAR-1 aims to address the critical needs for safety alignment in LRMs. Specifically, we begin by integrating existing open-source safety datasets from diverse sources. Then, we curate safety policies to generate policy-grounded deliberative reasoning samples. Lastly, we apply a GPT-4o-based safety scoring system to select training examples aligned with best practices. Experimental results show that fine-tuning LRMs with STAR-1 leads to an average 40% improvement in safety performance across four benchmarks, while only incurring a marginal decrease (e.g., an average of 1.1%) in reasoning ability measured across five reasoning tasks. Extensive ablation studies further validate the importance of our design principles in constructing STAR-1 and analyze its efficacy across both LRMs and traditional LLMs. Our project page is https://ucsc-vlaa.github.io/STAR-1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01879v1">TransientTables: Evaluating LLMs' Reasoning on Temporally Evolving Semi-structured Tables</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 19 Pages. 21 Tables, 1 figure
    </div>
    <details class="paper-abstract">
      Humans continuously make new discoveries, and understanding temporal sequence of events leading to these breakthroughs is essential for advancing science and society. This ability to reason over time allows us to identify future steps and understand the effects of financial and political decisions on our lives. However, large language models (LLMs) are typically trained on static datasets, limiting their ability to perform effective temporal reasoning. To assess the temporal reasoning capabilities of LLMs, we present the TRANSIENTTABLES dataset, which comprises 3,971 questions derived from over 14,000 tables, spanning 1,238 entities across multiple time periods. We introduce a template-based question-generation pipeline that harnesses LLMs to refine both templates and questions. Additionally, we establish baseline results using state-of-the-art LLMs to create a benchmark. We also introduce novel modeling strategies centered around task decomposition, enhancing LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06289v2">Automate Strategy Finding with LLM in Quant Investment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      Despite significant progress in deep learning for financial trading, existing models often face instability and high uncertainty, hindering their practical application. Leveraging advancements in Large Language Models (LLMs) and multi-agent architectures, we propose a novel framework for quantitative stock investment in portfolio management and alpha mining. Our framework addresses these issues by integrating LLMs to generate diversified alphas and employing a multi-agent approach to dynamically evaluate market conditions. This paper proposes a framework where large language models (LLMs) mine alpha factors from multimodal financial data, ensuring a comprehensive understanding of market dynamics. The first module extracts predictive signals by integrating numerical data, research papers, and visual charts. The second module uses ensemble learning to construct a diverse pool of trading agents with varying risk preferences, enhancing strategy performance through a broader market analysis. In the third module, a dynamic weight-gating mechanism selects and assigns weights to the most relevant agents based on real-time market conditions, enabling the creation of an adaptive and context-aware composite alpha formula. Extensive experiments on the Chinese stock markets demonstrate that this framework significantly outperforms state-of-the-art baselines across multiple financial metrics. The results underscore the efficacy of combining LLM-generated alphas with a multi-agent architecture to achieve superior trading performance and stability. This work highlights the potential of AI-driven approaches in enhancing quantitative investment strategies and sets a new benchmark for integrating advanced machine learning techniques in financial trading can also be applied on diverse markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v5">Non-Determinism of "Deterministic" LLM Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-02
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at https://github.com/breckbaldwin/llm-stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02017v1">Enhancing LLMs in Long Code Translation through Instrumentation and Program State Alignment</a></div>
    <div class="paper-meta">
      📅 2025-04-02
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Code translation aims to transform code between programming languages while preserving functionality, with applications in cross-platform development and software migration. Recent advances in Large Language Models (LLMs) have improved code translation, but challenges remain, particularly in inferring program functionality. These issues worsen with longer and more complex code, where current LLMs struggle to handle length and intricate semantics. To evaluate LLMs on long code translation, we introduce LongTrans, a large-scale execution-based benchmark with C++, Java, and Python programs, ranging from hundreds to thousands of tokens. Our empirical study of 12 LLMs reveals a sharp performance decline as code length increases, with even the best-performing model, GPT-4o, achieving only 57.51% computational accuracy. This highlights the need for further research in long code translation. We argue that code translation should maintain invariant functionality while transforming syntax and keywords across languages. Despite differences in appearance, program states should remain consistent throughout execution. To address this, we propose PAST (Program State Alignment augmented Translation), which integrates instrumentation to capture and align program states during translation. This approach is the first to leverage LLMs to insert instrumentation in both original and translated code, tracing program states at runtime. By prompting the LLM to correct errors based on output traces, we mitigate inconsistencies and enhance translation accuracy. Experimental results show significant improvements, with computational accuracy rising from 57.51% to 84.70% for GPT-4o, 50.68% to 69.97% for Mistral-Large-2, and 52.45% to 76.43% for DeepSeek-Coder-V2. These improvements are consistent across models and datasets, with ablation studies confirming the benefits of instrumentation and state alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19178v2">UQABench: Evaluating User Embedding for Prompting LLMs in Personalized Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 3 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable success in natural language processing (NLP). In practical scenarios like recommendations, as users increasingly seek personalized experiences, it becomes crucial to incorporate user interaction history into the context of LLMs to enhance personalization. However, from a practical utility perspective, user interactions' extensive length and noise present challenges when used directly as text prompts. A promising solution is to compress and distill interactions into compact embeddings, serving as soft prompts to assist LLMs in generating personalized responses. Although this approach brings efficiency, a critical concern emerges: Can user embeddings adequately capture valuable information and prompt LLMs? To address this concern, we propose \name, a benchmark designed to evaluate the effectiveness of user embeddings in prompting LLMs for personalization. We establish a fair and standardized evaluation process, encompassing pre-training, fine-tuning, and evaluation stages. To thoroughly evaluate user embeddings, we design three dimensions of tasks: sequence understanding, action prediction, and interest perception. These evaluation tasks cover the industry's demands in traditional recommendation tasks, such as improving prediction accuracy, and its aspirations for LLM-based methods, such as accurately understanding user interests and enhancing the user experience. We conduct extensive experiments on various state-of-the-art methods for modeling user embeddings. Additionally, we reveal the scaling laws of leveraging user embeddings to prompt LLMs. The benchmark is available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v2">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06994v2">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern applications codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.68\% (upto 65\%) reduction in application binary size, and an average of 2.8\% performance improvement over the traditional profile-guided optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06559v2">Is Your LLM Secretly a World Model of the Internet? Model-Based Planning for Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 22 pages, 11 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Language agents based on large language models (LLMs) have demonstrated great promise in automating web-based tasks. Recent work has shown that incorporating advanced planning algorithms, e.g., tree search, is advantageous over reactive planning for web agents. However, unlike simulated sandbox environments, real-world environments such as the web are rife with irreversible actions. This undermines the feasibility of backtracking, a cornerstone of (tree) search. Overly relying on test-time search also hurts efficiency. We advocate model-based planning for web agents that employs a world model to simulate and deliberate over the outcome of each candidate action before committing to one. We systematically explore this paradigm by (1) Proposing a model-based planning framework, WebDreamer, which employs LLMs to serve as both world models and value functions; (2) Training specialized LLMs as world models with a scalable data synthesis pipeline. Empirical results demonstrate that WebDreamer achieves substantial performance improvements over reactive baselines. It is competitive, while being 4-5 times more efficient, with tree search in sandbox environments (VisualWebArena) and also works effectively on real-world websites (Online-Mind2Web and Mind2Web-Live). Furthermore, our trained world model, Dreamer-7B, performs comparable to GPT-4o, highlighting the potential of specialized world models for efficient and effective planning in complex web environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.15426v2">CodingTeachLLM: Empowering LLM's Coding Ability via AST Prior Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce CodingTeachLLM, a large language model (LLM) designed for coding teaching. Specially, we aim to enhance the coding ability of LLM and lead it to better teaching mode in education context. Thus, we propose an end-to-end prior-based three-phases supervised fine-tuned model, which is proved more competitive than traditional fine-tuning method. More specifically, our model realizes the structural disassembly and incremental guided output of educational knowledge. To this end, we robustify data classification of three types via a sampler and overlap estimation neural network, and inject the preprocessing datasets into pre-trained model in three batches for LORA fine-tuning. Then, we design a prior module couples system prompt, vector databases, and abstract syntax tree task segmentation. Finally, the compression method and regularization constraint are applied to the prior-based fine-tuned model, followed by text filter at the output end to obtain incremental guided results. Our model represents the first research effort to truly embody the tutor role with the features of abundant educational knowledge, step-by-step incremental guided outputs and non-disclosure of answers. Extensive experiments report that our model also achieves state-of-the-art in code abilities compared to open-source models, reaching an impressive 75.10% on the HumanEval (@pass 1) benchmark. Additionally, our model maintains strong conversational capabilities, with the 13B quantized version achieving scores of 56.34, 50.60, and 45.27 respectively on the MMLU, C-Eval, and AGIEval (5 shot) dialogue evaluation benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v4">Non-Determinism of "Deterministic" LLM Settings</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at http://github.com/REDACTED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01141v2">How Well do LLMs Compress Their Own Chain-of-Thought? A Token Complexity Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Chain-of-thought prompting has emerged as a powerful technique for enabling large language models (LLMs) to solve complex reasoning tasks. However, these reasoning chains can be verbose, raising concerns about efficiency. In response, recent works have sought to decrease response lengths through simple prompting strategies (e.g. 'be concise'). In this work, we conduct the first systematic study of the relationship between reasoning length and model performance across a diverse range of compression instructions (e.g. 'use 10 words or less' or 'remove all punctuation'). In doing so, we discover a universal tradeoff between reasoning length and accuracy that persists across even very distinct reasoning chains. We demonstrate that this tradeoff emerges from a sharp threshold behavior at the question level: each task has an intrinsic 'token complexity' - a minimal number of tokens required for successful problem-solving. We show how token complexity enables us to compute information-theoretic limits on the accuracy-compression tradeoff, and find that prompt-based compression strategies operate far from these theoretical limits. This suggests there may be significant room for improvement and our framework provides a benchmark to help researchers evaluate progress in reasoning efficiency. Our work also highlights the importance of adaptive compression -- giving shorter responses for easier questions -- and we show that token complexity is a useful tool for measuring this capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19326v2">Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12836v2">EditRoom: LLM-parameterized Graph Diffusion for Composable 3D Room Layout Editing</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Given the steep learning curve of professional 3D software and the time-consuming process of managing large 3D assets, language-guided 3D scene editing has significant potential in fields such as virtual reality, augmented reality, and gaming. However, recent approaches to language-guided 3D scene editing either require manual interventions or focus only on appearance modifications without supporting comprehensive scene layout changes. In response, we propose EditRoom, a unified framework capable of executing a variety of layout edits through natural language commands, without requiring manual intervention. Specifically, EditRoom leverages Large Language Models (LLMs) for command planning and generates target scenes using a diffusion-based method, enabling six types of edits: rotate, translate, scale, replace, add, and remove. To address the lack of data for language-guided 3D scene editing, we have developed an automatic pipeline to augment existing 3D scene synthesis datasets and introduced EditRoom-DB, a large-scale dataset with 83k editing pairs, for training and evaluation. Our experiments demonstrate that our approach consistently outperforms other baselines across all metrics, indicating higher accuracy and coherence in language-guided scene layout editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09577v2">Flash normalization: fast normalization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 7 pages, 8 figures
    </div>
    <details class="paper-abstract">
      RMSNorm is used by many LLMs such as Llama, Mistral, and OpenELM. This paper details FlashNorm, which is an exact but faster implementation of RMSNorm followed by linear layers. FlashNorm also speeds up Layer Normalization and its recently proposed replacement Dynamic Tanh (DyT) arXiv:2503.10622. See https://github.com/OpenMachine-ai/transformer-tricks for code and more transformer tricks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01241v1">Catastrophic Forgetting in LLMs: A Comparative Analysis Across Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced Natural Language Processing (NLP), particularly in Natural Language Understanding (NLU) tasks. As we progress toward an agentic world where LLM-based agents autonomously handle specialized tasks, it becomes crucial for these models to adapt to new tasks without forgetting previously learned information - a challenge known as catastrophic forgetting. This study evaluates the continual fine-tuning of various open-source LLMs with different parameter sizes (specifically models under 10 billion parameters) on key NLU tasks from the GLUE benchmark, including SST-2, MRPC, CoLA, and MNLI. By employing prompt engineering and task-specific adjustments, we assess and compare the models' abilities to retain prior knowledge while learning new tasks. Our results indicate that models such as Phi-3.5-mini exhibit minimal forgetting while maintaining strong learning capabilities, making them well-suited for continual learning environments. Additionally, models like Orca-2-7b and Qwen2.5-7B demonstrate impressive learning abilities and overall performance after fine-tuning. This work contributes to understanding catastrophic forgetting in LLMs and highlights prompting engineering to optimize model performance for continual learning scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15487v2">Multi-LLM Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      In this work, we propose a Multi-LLM summarization framework, and investigate two different multi-LLM strategies including centralized and decentralized. Our multi-LLM summarization framework has two fundamentally important steps at each round of conversation: generation and evaluation. These steps are different depending on whether our multi-LLM decentralized summarization is used or centralized. In both our multi-LLM decentralized and centralized strategies, we have k different LLMs that generate diverse summaries of the text. However, during evaluation, our multi-LLM centralized summarization approach leverages a single LLM to evaluate the summaries and select the best one whereas k LLMs are used for decentralized multi-LLM summarization. Overall, we find that our multi-LLM summarization approaches significantly outperform the baselines that leverage only a single LLM by up to 3x. These results indicate the effectiveness of multi-LLM approaches for summarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01234v1">First Field-Trial Demonstration of L4 Autonomous Optical Network for Distributed AI Training Communication: An LLM-Powered Multi-AI-Agent Solution</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Submitted to the PDP session of the Optical Fiber Communications Conference (OFC) 2025
    </div>
    <details class="paper-abstract">
      We demonstrate the first cross-domain cross-layer level-4 autonomous optical network via a multi-AI-agent system. Field trials show 98 percent task completion rate across the distributed AI training lifecycle-3.2x higher than single agents using state-of-the-art LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01205v1">Epistemic Alignment: A Mediating Framework for User-LLM Knowledge Delivery</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      LLMs increasingly serve as tools for knowledge acquisition, yet users cannot effectively specify how they want information presented. When users request that LLMs "cite reputable sources," "express appropriate uncertainty," or "include multiple perspectives," they discover that current interfaces provide no structured way to articulate these preferences. The result is prompt sharing folklore: community-specific copied prompts passed through trust relationships rather than based on measured efficacy. We propose the Epistemic Alignment Framework, a set of ten challenges in knowledge transmission derived from the philosophical literature of epistemology, concerning issues such as evidence quality assessment and calibration of testimonial reliance. The framework serves as a structured intermediary between user needs and system capabilities, creating a common vocabulary to bridge the gap between what users want and what systems deliver. Through a thematic analysis of custom prompts and personalization strategies shared on online communities where these issues are actively discussed, we find users develop elaborate workarounds to address each of the challenges. We then apply our framework to two prominent model providers, OpenAI and Anthropic, through content analysis of their documented policies and product features. Our analysis shows that while these providers have partially addressed the challenges we identified, they fail to establish adequate mechanisms for specifying epistemic preferences, lack transparency about how preferences are implemented, and offer no verification tools to confirm whether preferences were followed. For AI developers, the Epistemic Alignment Framework offers concrete guidance for supporting diverse approaches to knowledge; for users, it works toward information delivery that aligns with their specific needs rather than defaulting to one-size-fits-all approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01145v1">MaLAware: Automating the Comprehension of Malicious Software Behaviours using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at MSR 2025
    </div>
    <details class="paper-abstract">
      Current malware (malicious software) analysis tools focus on detection and family classification but fail to provide clear and actionable narrative insights into the malignant activity of the malware. Therefore, there is a need for a tool that translates raw malware data into human-readable descriptions. Developing such a tool accelerates incident response, reduces malware analysts' cognitive load, and enables individuals having limited technical expertise to understand malicious software behaviour. With this objective, we present MaLAware, which automatically summarizes the full spectrum of malicious activity of malware executables. MaLAware processes Cuckoo Sandbox-generated reports using large language models (LLMs) to correlate malignant activities and generate concise summaries explaining malware behaviour. We evaluate the tool's performance on five open-source LLMs. The evaluation uses the human-written malware behaviour description dataset as ground truth. The model's performance is measured using 11 extensive performance metrics, which boosts the confidence of MaLAware's effectiveness. The current version of the tool, i.e., MaLAware, supports Qwen2.5-7B, Llama2-7B, Llama3.1-8B, Mistral-7B, and Falcon-7B, along with the quantization feature for resource-constrained environments. MaLAware lays a foundation for future research in malware behavior explanation, and its extensive evaluation demonstrates LLMs' ability to narrate malware behavior in an actionable and comprehensive manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.05396v2">FairCoder: Evaluating Social Bias of LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed in coding tasks, drawing increasing attention to the evaluation of the quality and safety of LLMs' outputs. However, research on bias in code generation remains limited. Existing studies typically identify bias by applying malicious prompts or reusing tasks and dataset originally designed for discriminative models. Given that prior datasets are not fully optimized for code-related tasks, there is a pressing need for benchmarks specifically designed for evaluating code models. In this study, we introduce FairCoder, a novel benchmark for evaluating social bias in code generation. FairCoder explores the bias issue following the pipeline in software development, from function implementation to unit test, with diverse real-world scenarios. Additionally, three metrics are designed to assess fairness performance on this benchmark. We conduct experiments on widely used LLMs and provide a comprehensive analysis of the results. The findings reveal that all tested LLMs exhibit social bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01127v1">Can LLMs Grasp Implicit Cultural Values? Benchmarking LLMs' Metacognitive Cultural Intelligence with CQ-Bench</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Cultural Intelligence (CQ) refers to the ability to understand unfamiliar cultural contexts-a crucial skill for large language models (LLMs) to effectively engage with globally diverse users. While existing research often focuses on explicitly stated cultural norms, such approaches fail to capture the subtle, implicit values that underlie real-world conversations. To address this gap, we introduce CQ-Bench, a benchmark specifically designed to assess LLMs' capability to infer implicit cultural values from natural conversational contexts. We generate a multi-character conversation-based stories dataset using values from the World Value Survey and GlobalOpinions datasets, with topics including ethical, religious, social, and political. Our dataset construction pipeline includes rigorous validation procedures-incorporation, consistency, and implicitness checks-using GPT-4o, with 98.2% human-model agreement in the final validation. Our benchmark consists of three tasks of increasing complexity: attitude detection, value selection, and value extraction. We find that while o1 and Deepseek-R1 models reach human-level performance in value selection (0.809 and 0.814), they still fall short in nuanced attitude detection, with F1 scores of 0.622 and 0.635, respectively. In the value extraction task, GPT-4o-mini and o3-mini score 0.602 and 0.598, highlighting the difficulty of open-ended cultural reasoning. Notably, fine-tuning smaller models (e.g., LLaMA-3.2-3B) on only 500 culturally rich examples improves performance by over 10%, even outperforming stronger baselines (o3-mini) in some cases. Using CQ-Bench, we provide insights into the current challenges in LLMs' CQ research and suggest practical pathways for enhancing LLMs' cross-cultural reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01094v1">Multilingual and Multi-Accent Jailbreaking of Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 21 pages, 6 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01005v1">When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00993v1">MedReason: Eliciting Factual Medical Reasoning Steps in LLMs via Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Medical tasks such as diagnosis and treatment planning require precise and complex reasoning, particularly in life-critical domains. Unlike mathematical reasoning, medical reasoning demands meticulous, verifiable thought processes to ensure reliability and accuracy. However, there is a notable lack of datasets that provide transparent, step-by-step reasoning to validate and enhance the medical reasoning ability of AI models. To bridge this gap, we introduce MedReason, a large-scale high-quality medical reasoning dataset designed to enable faithful and explainable medical problem-solving in large language models (LLMs). We utilize a structured medical knowledge graph (KG) to convert clinical QA pairs into logical chains of reasoning, or ``thinking paths'', which trace connections from question elements to answers via relevant KG entities. Each path is validated for consistency with clinical logic and evidence-based medicine. Our pipeline generates detailed reasoning for various medical questions from 7 medical datasets, resulting in a dataset of 32,682 question-answer pairs, each with detailed, step-by-step explanations. Experiments demonstrate that fine-tuning with our dataset consistently boosts medical problem-solving capabilities, achieving significant gains of up to 7.7% for DeepSeek-Ditill-8B. Our top-performing model, MedReason-8B, outperforms the Huatuo-o1-8B, a state-of-the-art medical reasoning model, by up to 4.2% on the clinical benchmark MedBullets. We also engage medical professionals from diverse specialties to assess our dataset's quality, ensuring MedReason offers accurate and coherent medical reasoning. Our data, models, and code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00858v1">Whispering Under the Eaves: Protecting User Privacy Against Commercial and LLM-powered Automatic Speech Recognition Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accept to USENIX Security 2025
    </div>
    <details class="paper-abstract">
      The widespread application of automatic speech recognition (ASR) supports large-scale voice surveillance, raising concerns about privacy among users. In this paper, we concentrate on using adversarial examples to mitigate unauthorized disclosure of speech privacy thwarted by potential eavesdroppers in speech communications. While audio adversarial examples have demonstrated the capability to mislead ASR models or evade ASR surveillance, they are typically constructed through time-intensive offline optimization, restricting their practicality in real-time voice communication. Recent work overcame this limitation by generating universal adversarial perturbations (UAPs) and enhancing their transferability for black-box scenarios. However, they introduced excessive noise that significantly degrades audio quality and affects human perception, thereby limiting their effectiveness in practical scenarios. To address this limitation and protect live users' speech against ASR systems, we propose a novel framework, AudioShield. Central to this framework is the concept of Transferable Universal Adversarial Perturbations in the Latent Space (LS-TUAP). By transferring the perturbations to the latent space, the audio quality is preserved to a large extent. Additionally, we propose target feature adaptation to enhance the transferability of UAPs by embedding target text features into the perturbations. Comprehensive evaluation on four commercial ASR APIs (Google, Amazon, iFlytek, and Alibaba), three voice assistants, two LLM-powered ASR and one NN-based ASR demonstrates the protection superiority of AudioShield over existing competitors, and both objective and subjective evaluations indicate that AudioShield significantly improves the audio quality. Moreover, AudioShield also shows high effectiveness in real-time end-to-end scenarios, and demonstrates strong resilience against adaptive countermeasures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00829v1">How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of Large Language Models (LLMs) with efficiency and scalability remains a fundamental challenge in artificial intelligence research. This paper presents a rigorous experimental investigation into how difficulty-aware staged reinforcement learning (RL) strategies can substantially improve LLM reasoning performance. Through systematic analysis, we demonstrate that strategically selecting training data according to well-defined difficulty levels markedly enhances RL optimization. Moreover, we introduce a staged training methodology, progressively exposing models to increasingly challenging tasks, further amplifying reasoning capabilities. Our findings reveal significant cross-domain benefits when simultaneously training models on mathematical reasoning and code generation tasks. Notably, our proposed approach enables a 1.5B parameter model to achieve an accuracy of 42.3\% on the AIME-2024 benchmark, 89.5\% on the MATH-500 benchmark. These results underscore the efficacy of our method in advancing the reasoning proficiency of LLMs. We will open-source our datasets on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00727v1">Personality-Driven Decision-Making in LLM-Based Autonomous Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 8 figures. To be included in Proc. of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2025)
    </div>
    <details class="paper-abstract">
      The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00711v1">GraphMaster: Automated Graph Synthesis via LLM Agents in Data-Limited Environments</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The era of foundation models has revolutionized AI research, yet Graph Foundation Models (GFMs) remain constrained by the scarcity of large-scale graph corpora. Traditional graph data synthesis techniques primarily focus on simplistic structural operations, lacking the capacity to generate semantically rich nodes with meaningful textual attributes: a critical limitation for real-world applications. While large language models (LLMs) demonstrate exceptional text generation capabilities, their direct application to graph synthesis is impeded by context window limitations, hallucination phenomena, and structural consistency challenges. To address these issues, we introduce GraphMaster, the first multi-agent framework specifically designed for graph data synthesis in data-limited environments. GraphMaster orchestrates four specialized LLM agents (Manager, Perception, Enhancement, and Evaluation) that collaboratively optimize the synthesis process through iterative refinement, ensuring both semantic coherence and structural integrity. To rigorously evaluate our approach, we create new data-limited "Sub" variants of six standard graph benchmarks, specifically designed to test synthesis capabilities under realistic constraints. Additionally, we develop a novel interpretability assessment framework that combines human evaluation with a principled Grassmannian manifold-based analysis, providing both qualitative and quantitative measures of semantic coherence. Experimental results demonstrate that GraphMaster significantly outperforms traditional synthesis methods across multiple datasets, establishing a strong foundation for advancing GFMs in data-scarce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00694v1">On Benchmarking Code LLMs for Android Malware Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in various code intelligence tasks. However, their effectiveness for Android malware analysis remains underexplored. Decompiled Android code poses unique challenges for analysis, primarily due to its large volume of functions and the frequent absence of meaningful function names. This paper presents Cama, a benchmarking framework designed to systematically evaluate the effectiveness of Code LLMs in Android malware analysis tasks. Cama specifies structured model outputs (comprising function summaries, refined function names, and maliciousness scores) to support key malware analysis tasks, including malicious function identification and malware purpose summarization. Built on these, it integrates three domain-specific evaluation metrics, consistency, fidelity, and semantic relevance, enabling rigorous stability and effectiveness assessment and cross-model comparison. We construct a benchmark dataset consisting of 118 Android malware samples, encompassing over 7.5 million distinct functions, and use Cama to evaluate four popular open-source models. Our experiments provide insights into how Code LLMs interpret decompiled code and quantify the sensitivity to function renaming, highlighting both the potential and current limitations of Code LLMs in malware analysis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00664v1">Do LLMs Surpass Encoders for Biomedical NER?</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted to appear in IEEE ICHI 2025
    </div>
    <details class="paper-abstract">
      Recognizing spans of biomedical concepts and their types (e.g., drug or gene) in free text, often called biomedical named entity recognition (NER), is a basic component of information extraction (IE) pipelines. Without a strong NER component, other applications, such as knowledge discovery and information retrieval, are not practical. State-of-the-art in NER shifted from traditional ML models to deep neural networks with transformer-based encoder models (e.g., BERT) emerging as the current standard. However, decoder models (also called large language models or LLMs) are gaining traction in IE. But LLM-driven NER often ignores positional information due to the generative nature of decoder models. Furthermore, they are computationally very expensive (both in inference time and hardware needs). Hence, it is worth exploring if they actually excel at biomedical NER and assess any associated trade-offs (performance vs efficiency). This is exactly what we do in this effort employing the same BIO entity tagging scheme (that retains positional information) using five different datasets with varying proportions of longer entities. Our results show that the LLMs chosen (Mistral and Llama: 8B range) often outperform best encoder models (BERT-(un)cased, BiomedBERT, and DeBERTav3: 300M range) by 2-8% in F-scores except for one dataset, where they equal encoder performance. This gain is more prominent among longer entities of length >= 3 tokens. However, LLMs are one to two orders of magnitude more expensive at inference time and may need cost prohibitive hardware. Thus, when performance differences are small or real time user feedback is needed, encoder models might still be more suitable than LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00636v1">Exploring the Impact of an LLM-Powered Teachable Agent on Learning Gains and Cognitive Load in Music Education</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted at CHI 2025 Workshop on Augmented Educators and AI: Shaping the Future of Human and AI Cooperation in Learning
    </div>
    <details class="paper-abstract">
      This study examines the impact of an LLM-powered teachable agent, grounded in the Learning by Teaching (LBT) pedagogy, on students' music theory learning and cognitive load. The participants were 28 Chinese university students with prior music instrumental experiences. In an online experiment, they were assigned to either an experimental group, which engaged in music analysis with the teachable agent, or a control group, which conducted self-directed analysis using instructional materials. Findings indicate that students in the experimental group achieved significantly higher post-test scores than those in the control group. Additionally, they reported lower cognitive load, suggesting that the teachable agent effectively reduced the cognitive demands of music analysis tasks. These results highlight the potential of AI-driven scaffolding based on LBT principles to enhance music theory education, supporting teachers in delivering theory-oriented instruction while fostering students' self-directed learning skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00613v1">LLM-Guided Search for Deletion-Correcting Codes</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Finding deletion-correcting codes of maximum size has been an open problem for over 70 years, even for a single deletion. In this paper, we propose a novel approach for constructing deletion-correcting codes. A code is a set of sequences satisfying certain constraints, and we construct it by greedily adding the highest-priority sequence according to a priority function. To find good priority functions, we leverage FunSearch, a large language model (LLM)-guided evolutionary search proposed by Romera et al., 2024. FunSearch iteratively generates, evaluates, and refines priority functions to construct large deletion-correcting codes. For a single deletion, our evolutionary search finds functions that construct codes which match known maximum sizes, reach the size of the largest (conjectured optimal) Varshamov-Tenengolts codes where the maximum is unknown, and independently rediscover them in equivalent form. For two deletions, we find functions that construct codes with new best-known sizes for code lengths \( n = 12, 13 \), and \( 16 \), establishing improved lower bounds. These results demonstrate the potential of LLM-guided search for information theory and code design and represent the first application of such methods for constructing error-correcting codes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00607v1">Contextualized Autonomous Drone Navigation using LLMs Deployed in Edge-Cloud Computing</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Autonomous navigation is usually trained offline in diverse scenarios and fine-tuned online subject to real-world experiences. However, the real world is dynamic and changeable, and many environmental encounters/effects are not accounted for in real-time due to difficulties in describing them within offline training data or hard to describe even in online scenarios. However, we know that the human operator can describe these dynamic environmental encounters through natural language, adding semantic context. The research is to deploy Large Language Models (LLMs) to perform real-time contextual code adjustment to autonomous navigation. The challenge not evaluated in literature is what LLMs are appropriate and where should these computationally heavy algorithms sit in the computation-communication edge-cloud computing architectures. In this paper, we evaluate how different LLMs can adjust both the navigation map parameters dynamically (e.g., contour map shaping) and also derive navigation task instruction sets. We then evaluate which LLMs are most suitable and where they should sit in future edge-cloud of 6G telecommunication architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00587v1">AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has catalyzed the development of multi-agent systems, where multiple LLM-based agents collaborate to solve complex tasks. However, existing systems predominantly rely on centralized coordination, which introduces scalability bottlenecks, limits adaptability, and creates single points of failure. Additionally, concerns over privacy and proprietary knowledge sharing hinder cross-organizational collaboration, leading to siloed expertise. To address these challenges, we propose AgentNet, a decentralized, Retrieval-Augmented Generation (RAG)-based framework that enables LLM-based agents to autonomously evolve their capabilities and collaborate efficiently in a Directed Acyclic Graph (DAG)-structured network. Unlike traditional multi-agent systems that depend on static role assignments or centralized control, AgentNet allows agents to specialize dynamically, adjust their connectivity, and route tasks without relying on predefined workflows. AgentNet's core design is built upon several key innovations: (1) Fully Decentralized Paradigm: Removing the central orchestrator, allowing agents to coordinate and specialize autonomously, fostering fault tolerance and emergent collective intelligence. (2) Dynamically Evolving Graph Topology: Real-time adaptation of agent connections based on task demands, ensuring scalability and resilience.(3) Adaptive Learning for Expertise Refinement: A retrieval-based memory system that enables agents to continuously update and refine their specialized skills. By eliminating centralized control, AgentNet enhances fault tolerance, promotes scalable specialization, and enables privacy-preserving collaboration across organizations. Through decentralized coordination and minimal data exchange, agents can leverage diverse knowledge sources while safeguarding sensitive information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v1">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00428v1">LLM-Assisted Proactive Threat Intelligence for Automated Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 Pages, 1 Figure
    </div>
    <details class="paper-abstract">
      Successful defense against dynamically evolving cyber threats requires advanced and sophisticated techniques. This research presents a novel approach to enhance real-time cybersecurity threat detection and response by integrating large language models (LLMs) and Retrieval-Augmented Generation (RAG) systems with continuous threat intelligence feeds. Leveraging recent advancements in LLMs, specifically GPT-4o, and the innovative application of RAG techniques, our approach addresses the limitations of traditional static threat analysis by incorporating dynamic, real-time data sources. We leveraged RAG to get the latest information in real-time for threat intelligence, which is not possible in the existing GPT-4o model. We employ the Patrowl framework to automate the retrieval of diverse cybersecurity threat intelligence feeds, including Common Vulnerabilities and Exposures (CVE), Common Weakness Enumeration (CWE), Exploit Prediction Scoring System (EPSS), and Known Exploited Vulnerabilities (KEV) databases, and integrate these with the all-mpnet-base-v2 model for high-dimensional vector embeddings, stored and queried in Milvus. We demonstrate our system's efficacy through a series of case studies, revealing significant improvements in addressing recently disclosed vulnerabilities, KEVs, and high-EPSS-score CVEs compared to the baseline GPT-4o. This work not only advances the role of LLMs in cybersecurity but also establishes a robust foundation for the development of automated intelligent cyberthreat information management systems, addressing crucial gaps in current cybersecurity practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00409v1">Semantic Mastery: Enhancing LLMs with Advanced Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have greatly improved their capability in performing NLP tasks. However, deeper semantic understanding, contextual coherence, and more subtle reasoning are still difficult to obtain. The paper discusses state-of-the-art methodologies that advance LLMs with more advanced NLU techniques, such as semantic parsing, knowledge integration, and contextual reinforcement learning. We analyze the use of structured knowledge graphs, retrieval-augmented generation (RAG), and fine-tuning strategies that match models with human-level understanding. Furthermore, we address the incorporation of transformer-based architectures, contrastive learning, and hybrid symbolic-neural methods that address problems like hallucinations, ambiguity, and inconsistency in the factual perspectives involved in performing complex NLP tasks, such as question-answering text summarization and dialogue generation. Our findings show the importance of semantic precision for enhancing AI-driven language systems and suggest future research directions to bridge the gap between statistical language models and true natural language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00374v1">When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR)</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00341v1">Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 This article has been accepted for publication in the IEEE 2025 International Conference on Communications (ICC2025)
    </div>
    <details class="paper-abstract">
      The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00310v1">Detecting and Mitigating Bias in LLMs through Knowledge Graph-Augmented Training</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Large language models have revolutionized natural language processing with their surprising capability to understand and generate human-like text. However, many of these models inherit and further amplify the biases present in their training data, raising ethical and fairness concerns. The detection and mitigation of such biases are vital to ensuring that LLMs act responsibly and equitably across diverse domains. This work investigates Knowledge Graph-Augmented Training (KGAT) as a novel method to mitigate bias in LLM. Using structured domain-specific knowledge from real-world knowledge graphs, we improve the understanding of the model and reduce biased output. Public datasets for bias assessment include Gender Shades, Bias in Bios, and FairFace, while metrics such as demographic parity and equal opportunity facilitate rigorous detection. We also performed targeted mitigation strategies to correct biased associations, leading to a significant drop in biased output and improved bias metrics. Equipped with real-world datasets and knowledge graphs, our framework is both scalable and effective, paving the way toward responsible deployment in sensitive and high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01995v1">Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shown impressive progress in mathematical reasoning tasks. However, current evaluation benchmarks predominantly focus on the accuracy of final answers, often overlooking the logical rigor crucial for mathematical problem-solving. The claim that state-of-the-art LLMs can solve Math Olympiad-level problems requires closer examination. To explore this, we conducted both qualitative and quantitative human evaluations of proofs generated by LLMs, and developed a schema for automatically assessing their reasoning capabilities. Our study reveals that current LLMs fall significantly short of solving challenging Olympiad-level problems and frequently fail to distinguish correct mathematical reasoning from clearly flawed solutions. We also found that occasional correct final answers provided by LLMs often result from pattern recognition or heuristic shortcuts rather than genuine mathematical reasoning. These findings underscore the substantial gap between LLM performance and human expertise in advanced mathematical reasoning and highlight the importance of developing benchmarks that prioritize the rigor and coherence of mathematical arguments rather than merely the correctness of final answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02873v1">Short-PHD: Detecting Short LLM-generated Text with Topological Data Analysis After Off-topic Content Insertion</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      The malicious usage of large language models (LLMs) has motivated the detection of LLM-generated texts. Previous work in topological data analysis shows that the persistent homology dimension (PHD) of text embeddings can serve as a more robust and promising score than other zero-shot methods. However, effectively detecting short LLM-generated texts remains a challenge. This paper presents Short-PHD, a zero-shot LLM-generated text detection method tailored for short texts. Short-PHD stabilizes the estimation of the previous PHD method for short texts by inserting off-topic content before the given input text and identifies LLM-generated text based on an established detection threshold. Experimental results on both public and generated datasets demonstrate that Short-PHD outperforms existing zero-shot methods in short LLM-generated text detection. Implementation codes are available online.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02870v1">AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Accepted by CVPR 2025 Workshop
    </div>
    <details class="paper-abstract">
      Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02867v1">Multi-Agent LLM Judge: automatic personalized LLM judge design for evaluating natural language generation applications</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Presented at SophiaSummit2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance across diverse domains, yet they still encounter challenges such as insufficient domain-specific knowledge, biases, and hallucinations. This underscores the need for robust evaluation methodologies to accurately assess LLM-based applications. Traditional evaluation methods, which rely on word overlap or text embeddings, are inadequate for capturing the nuanced semantic information necessary to evaluate dynamic, open-ended text generation. Recent research has explored leveraging LLMs to mimic human reasoning and decision-making processes for evaluation purposes known as LLM-as-a-judge framework. However, these existing frameworks have two significant limitations. First, they lack the flexibility to adapt to different text styles, including various answer and ground truth styles, thereby reducing their generalization performance. Second, the evaluation scores produced by these frameworks are often skewed and hard to interpret, showing a low correlation with human judgment. To address these challenges, we propose a novel dynamic multi-agent system that automatically designs personalized LLM judges for various natural language generation applications. This system iteratively refines evaluation prompts and balances the trade-off between the adaptive requirements of downstream tasks and the alignment with human perception. Our experimental results show that the proposed multi-agent LLM Judge framework not only enhances evaluation accuracy compared to existing methods but also produces evaluation scores that better align with human perception.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v3">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13727v2">LLM-Human Pipeline for Cultural Context Grounding of Conversations</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Oral at NAACL 2025 Main conference. Albuquerque, USA. Apr 29 - May 4, 2025. 19 pages, 9 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Conversations often adhere to well-understood social norms that vary across cultures. For example, while "addressing parents by name" is commonplace in the West, it is rare in most Asian cultures. Adherence or violation of such norms often dictates the tenor of conversations. Humans are able to navigate social situations requiring cultural awareness quite adeptly. However, it is a hard task for NLP models. In this paper, we tackle this problem by introducing a "Cultural Context Schema" for conversations. It comprises (1) conversational information such as emotions, dialogue acts, etc., and (2) cultural information such as social norms, violations, etc. We generate ~110k social norm and violation descriptions for ~23k conversations from Chinese culture using LLMs. We refine them using automated verification strategies which are evaluated against culturally aware human judgements. We organize these descriptions into meaningful structures we call "Norm Concepts", using an interactive human-in-loop framework. We ground the norm concepts and the descriptions in conversations using symbolic annotation. Finally, we use the obtained dataset for downstream tasks such as emotion, sentiment, and dialogue act detection. We show that it significantly improves the empirical performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14642v2">TOMG-Bench: Evaluating LLMs on Text-based Open Molecule Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 The first benchmark for text-based open molecule generation
    </div>
    <details class="paper-abstract">
      In this paper, we propose Text-based Open Molecule Generation Benchmark (TOMG-Bench), the first benchmark to evaluate the open-domain molecule generation capability of LLMs. TOMG-Bench encompasses a dataset of three major tasks: molecule editing (MolEdit), molecule optimization (MolOpt), and customized molecule generation (MolCustom). Each major task further contains three subtasks, while each subtask comprises 5,000 test samples. Given the inherent complexity of open molecule generation evaluation, we also developed an automated evaluation system that helps measure both the quality and the accuracy of the generated molecules. Our comprehensive benchmarking of 25 LLMs reveals the current limitations as well as potential areas for improvement in text-guided molecule discovery. Furthermore, we propose OpenMolIns, a specialized instruction tuning dataset established for solving challenges raised by TOMG-Bench. Fine-tuned on OpenMolIns, Llama3.1-8B could outperform all the open-source general LLMs, even surpassing GPT-3.5-turbo by 46.5\% on TOMG-Bench. Our codes and datasets are available through https://github.com/phenixace/TOMG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15035v2">LLMs Lost in Translation: M-ALERT uncovers Cross-Linguistic Safety Gaps</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Building safe Large Language Models (LLMs) across multiple languages is essential in ensuring both safe access and linguistic diversity. To this end, we introduce M-ALERT, a multilingual benchmark that evaluates the safety of LLMs in five languages: English, French, German, Italian, and Spanish. M-ALERT includes 15k high-quality prompts per language, totaling 75k, following the detailed ALERT taxonomy. Our extensive experiments on 10 state-of-the-art LLMs highlight the importance of language-specific safety analysis, revealing that models often exhibit significant inconsistencies in safety across languages and categories. For instance, Llama3.2 shows high unsafety in the category crime_tax for Italian but remains safe in other languages. Similar differences can be observed across all models. In contrast, certain categories, such as substance_cannabis and crime_propaganda, consistently trigger unsafe responses across models and languages. These findings underscore the need for robust multilingual safety practices in LLMs to ensure safe and responsible usage across diverse user communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13543v2">BALROG: Benchmarking Agentic LLM and VLM Reasoning On Games</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and Vision Language Models (VLMs) possess extensive knowledge and exhibit promising reasoning abilities, however, they still struggle to perform well in complex, dynamic environments. Real-world tasks require handling intricate interactions, advanced spatial reasoning, long-term planning, and continuous exploration of new strategies-areas in which we lack effective methodologies for comprehensively evaluating these capabilities. To address this gap, we introduce BALROG, a novel benchmark designed to assess the agentic capabilities of LLMs and VLMs through a diverse set of challenging games. Our benchmark incorporates a range of existing reinforcement learning environments with varying levels of difficulty, including tasks that are solvable by non-expert humans in seconds to extremely challenging ones that may take years to master (e.g., the NetHack Learning Environment). We devise fine-grained metrics to measure performance and conduct an extensive evaluation of several popular open-source and closed-source LLMs and VLMs. Our findings indicate that while current models achieve partial success in the easier games, they struggle significantly with more challenging tasks. Notably, we observe severe deficiencies in vision-based decision-making, as several models perform worse when visual representations of the environments are provided. We release BALROG as an open and user-friendly benchmark to facilitate future research and development in the agentic community. Code and Leaderboard at balrogai.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05447v2">TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is one of the leading and most widely used techniques for enhancing LLM retrieval capabilities, but it still faces significant limitations in commercial use cases. RAG primarily relies on the query-chunk text-to-text similarity in the embedding space for retrieval and can fail to capture deeper semantic relationships across chunks, is highly sensitive to chunking strategies, and is prone to hallucinations. To address these challenges, we propose TOBUGraph, a graph-based retrieval framework that first constructs the knowledge graph from unstructured data dynamically and automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse relationships among data, going beyond RAG's text-to-text similarity. Retrieval is achieved through graph traversal, leveraging the extracted relationships and structures to enhance retrieval accuracy, eliminating the need for chunking configurations while reducing hallucination. We demonstrate TOBUGraph's effectiveness in TOBU, a real-world application in production for personal memory organization and retrieval. Our evaluation using real user data demonstrates that TOBUGraph outperforms multiple RAG implementations in both precision and recall, significantly improving user experience through improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06621v4">Towards Robust and Parameter-Efficient Knowledge Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 ICLR 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong reasoning and memorization capabilities via pretraining on massive textual corpora. However, this poses risk of privacy and copyright violations, highlighting the need for efficient machine unlearning methods that remove sensitive data without retraining from scratch. While Gradient Ascent (GA) is commonly used to unlearn by reducing the likelihood of generating unwanted content, it leads to unstable optimization and catastrophic forgetting of retrained knowledge. We find that combining GA with low-rank adaptation results in poor trade-offs between computational cost and generative performance. To address these challenges, we propose Low-rank Knowledge Unlearning (LoKU), a novel framework that enables robust and efficient unlearning for LLMs. First, we introduce Inverted Hinge Loss, which suppresses unwanted tokens while maintaining fluency by boosting the probability of the next most likely token. Second, we develop a data-adaptive initialization for LoRA adapters via low-rank approximation weighted with relative Fisher information, thereby focusing updates on parameters critical for removing targeted knowledge. Experiments on the Training Data Extraction Challenge dataset using GPT-Neo models as well as on the TOFU benchmark with Phi-1.5B and Llama2-7B models demonstrate that our approach effectively removes sensitive information while maintaining reasoning and generative capabilities with minimal impact. Our implementation can be found in https://github.com/csm9493/efficient-llm-unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09078v5">Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable abilities across various language tasks, but solving complex reasoning problems remains a significant challenge. While existing methods, such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT), enhance reasoning by decomposing problems or structuring prompts, they typically perform a single pass of reasoning and may fail to revisit flawed paths, compromising accuracy. To address this limitation, we propose a novel reasoning framework called Forest-of-Thought (FoT), which integrates multiple reasoning trees to leverage collective decision-making for solving complex logical problems. FoT employs sparse activation strategies to select the most relevant reasoning paths, improving both efficiency and accuracy. Additionally, we introduce a dynamic self-correction strategy that enables real-time error correction, along with consensus-guided decision-making strategies to optimize both correctness and computational resources. Experimental results demonstrate that the FoT framework, combined with these strategies, significantly enhances the reasoning capabilities of LLMs, enabling them to solve complex tasks with greater precision and efficiency. Code will be available at https://github.com/iamhankai/Forest-of-Thought.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22968v2">HRET: A Self-Evolving LLM Evaluation Toolkit for Korean</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      Recent advancements in Korean large language models (LLMs) have spurred numerous benchmarks and evaluation methodologies, yet the lack of a standardized evaluation framework has led to inconsistent results and limited comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an open-source, self-evolving evaluation framework tailored specifically for Korean LLMs. HRET unifies diverse evaluation methods, including logit-based scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge assessments. Its modular, registry-based architecture integrates major benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends (vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for continuous evolution, HRET provides a robust foundation for reproducible, fair, and transparent Korean NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v3">A Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-04-01
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
